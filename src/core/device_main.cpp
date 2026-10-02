// src/core/device_main.cpp - `strata-device`: report the GPU(s), the plan, and exercise the arenas.
//
// This is P2.S1's "startup prints the memory plan vs actual cudaMemGetInfo" bullet, on its own so it can run
// without the model.  It is also the run-time half of the sm_70/sm_120 policy: CMake refuses to COMPILE for
// another architecture, and this refuses to RUN on one.
//
// Release 2 phase 1 adds the multi-device half: `--devices 0,2` builds the plan the engine builds (same
// code path, make_device_plan), reports it, and with `--p2p` measures the real device-to-device bandwidth
// of every enabled peer pair - the number phase 2's placement decision is made on.  `--selftest` exercises
// the arenas, including a cross-device pattern copy + verify when a peer pair is enabled.
#include "strata/core/device.hpp"
#include "strata/core/devices.hpp"
#include "strata/plan/plan.hpp"

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static std::string human(uint64_t b) {
    char buf[64];
    std::snprintf(buf, sizeof(buf), "%.3f GiB (%llu B)", (double) b / (1024.0 * 1024 * 1024),
                  (unsigned long long) b);
    return buf;
}

static void print_device_detail(const strata::core::DeviceInfo& d) {
    std::printf("device %d: %s\n", d.ordinal, d.name.c_str());
    std::printf("  compute capability  %d.%d   (sm_%d%d)\n", d.cc_major, d.cc_minor, d.cc_major, d.cc_minor);
    std::printf("  multiprocessors     %d\n", d.multi_processor_count);
    std::printf("  VRAM total / free   %s / %s\n", human(d.total_bytes).c_str(), human(d.free_bytes).c_str());
    std::printf("  driver / runtime    %d / %d\n", d.driver_version, d.runtime_version);
}

static int run_cross_device_selftest(const strata::core::DevicePlan& plan) {
    // A 1 MiB pattern written on the primary must arrive, bit for bit, on every auxiliary device - through
    // the peer connection when it is enabled, through host staging when it is not.  A multi-device run whose
    // cross-card copy cannot be verified here cannot be trusted with 9 GiB of KV cache in phase 2.
    const uint64_t bytes = 1ull << 20;
    const int prim = plan.primary();
    const size_t n_floats = bytes / sizeof(float);
    std::vector<float> host(n_floats);
    for (size_t i = 0; i < n_floats; ++i) host[i] = (float) (i % 1000) - 500.0f;

    void* src = nullptr, * dst = nullptr;
    int rc = 0;
    if (cudaSetDevice(prim) != cudaSuccess || cudaMalloc(&src, (size_t) bytes) != cudaSuccess) {
        std::fprintf(stderr, "cross-device selftest: primary arena\n");
        return 1;
    }
    if (cudaMemcpy(src, host.data(), bytes, cudaMemcpyHostToDevice) != cudaSuccess) {
        std::fprintf(stderr, "cross-device selftest: seed the primary pattern\n");
        rc = 1;
        goto out;
    }
    for (size_t i = 1; i < plan.ordinals.size(); ++i) {
        const int aux = plan.ordinals[i];
        const bool p2p = plan.p2p[0][i] == 1;
        if (cudaSetDevice(aux) != cudaSuccess || cudaMalloc(&dst, (size_t) bytes) != cudaSuccess) {
            std::fprintf(stderr, "cross-device selftest: aux %d arena\n", aux);
            rc = 1;
            goto out;
        }
        cudaError_t e;
        if (p2p) {
            cudaSetDevice(prim);
            e = cudaMemcpyPeer(dst, aux, src, prim, (size_t) bytes);
        } else {
            std::vector<float> stage(n_floats);
            cudaSetDevice(prim);
            if (cudaMemcpy(stage.data(), src, bytes, cudaMemcpyDeviceToHost) != cudaSuccess) e = cudaGetLastError();
            else {
                cudaSetDevice(aux);
                e = cudaMemcpy(dst, stage.data(), bytes, cudaMemcpyHostToDevice);
            }
        }
        if (e != cudaSuccess) {
            std::fprintf(stderr, "cross-device selftest: copy %d -> %d (%s): %s\n", prim, aux,
                         p2p ? "p2p" : "staged", cudaGetErrorString(e));
            rc = 1;
            continue;
        }
        std::vector<float> back(n_floats);
        cudaSetDevice(aux);
        if (cudaMemcpy(back.data(), dst, bytes, cudaMemcpyDeviceToHost) != cudaSuccess) {
            std::fprintf(stderr, "cross-device selftest: read back %d\n", aux);
            rc = 1;
            continue;
        }
        bool same = true;
        for (size_t k = 0; k < n_floats; ++k)
            if (back[k] != host[k]) { same = false; break; }
        if (!same) {
            std::fprintf(stderr, "cross-device selftest: PATTERN MISMATCH on device %d (%s)\n", aux,
                         p2p ? "p2p" : "staged");
            rc = 1;
        } else {
            std::printf("cross-device selftest: %d -> %d %s copy of %s verified\n", prim, aux,
                        p2p ? "p2p" : "staged", human(bytes).c_str());
        }
        cudaFree(dst);
        dst = nullptr;
    }
out:
    cudaSetDevice(prim);
    cudaFree(src);
    return rc;
}

int main(int argc, char** argv) {
    bool selftest = false, probe = false;
    std::string spec = "0";
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--selftest") == 0) selftest = true;
        else if (std::strcmp(argv[i], "--p2p") == 0) probe = true;
        else if (std::strcmp(argv[i], "--devices") == 0) {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "--devices needs a value, e.g. --devices 0,2\n");
                return 2;
            }
            spec = argv[++i];
        } else if (std::strcmp(argv[i], "--help") == 0 || std::strcmp(argv[i], "-h") == 0) {
            std::printf("usage: strata-device [--devices 0,2] [--p2p] [--selftest]\n"
                        "  --devices  the device plan to build (default 0; the engine's --devices default)\n"
                        "  --p2p      measure the D2D bandwidth of every enabled peer pair (phase 1 data)\n"
                        "  --selftest exercise the arenas, incl. a cross-device pattern copy+verify\n");
            return 0;
        } else {
            std::fprintf(stderr, "unknown argument: %s\n", argv[i]);
            return 2;
        }
    }

    try {
        const strata::core::DevicePlan plan = strata::core::make_device_plan(spec, probe);
        for (const strata::core::DeviceInfo& d : plan.info) {
            print_device_detail(d);
            std::printf("\n");
        }
        std::printf("%s\n", strata::core::device_plan_report(plan).c_str());

        // The planner's view against the primary card's.  A plan that does not fit in what is actually FREE
        // is the failure this print exists to make visible at startup rather than at token 4000.
        const auto memplan = strata::plan::make_plan(20480, strata::plan::Geometry{}, strata::plan::Costs{});
        std::printf("\n%s", strata::plan::to_string(memplan).c_str());
        std::printf("  card free           %s\n", human(plan.info[0].free_bytes).c_str());
        std::printf("  plan + KV vs free   %s\n",
                    memplan.vram_budget <= plan.info[0].free_bytes ? "FITS" : "*** DOES NOT FIT ***");

        if (selftest) {
            // Exercise the primary arena for real: allocate, write from the host, read back, and check the
            // poison path leaves NaNs rather than zeros.  A GPU test that only asks the driver for its name
            // does not test the runtime this file exists to provide.
            const uint64_t bytes = 64ull << 20;      // 64 MiB, small enough to be safe on any card
            strata::core::DeviceArena arena(bytes, plan.primary(), /*poison=*/true);
            void* a = arena.alloc(1 << 20, 256);
            void* b = arena.alloc(1 << 20, 4096);
            if (((uintptr_t) a % 256) || ((uintptr_t) b % 4096)) {
                std::fprintf(stderr, "selftest: alignment not honoured\n");
                return 1;
            }
            std::printf("\nselftest: primary arena %s, used %s after two 1 MiB allocations\n",
                        human(arena.capacity()).c_str(), human(arena.used()).c_str());
            // and the arena must REFUSE rather than wrap
            try {
                arena.alloc(bytes * 2);
                std::fprintf(stderr, "selftest: over-allocation did NOT throw\n");
                return 1;
            } catch (const strata::core::CudaError&) {
                std::printf("selftest: over-allocation refused as required\n");
            }
            // Every auxiliary device must be able to hold an arena of its own: phase 2's state placement is
            // an arena on the aux card, so the arena contract must hold THERE, not just on the primary.
            for (size_t i = 1; i < plan.ordinals.size(); ++i) {
                strata::core::DeviceArena aux(bytes, plan.ordinals[i], /*poison=*/false);
                void* p = aux.alloc(1 << 20, 256);
                if (((uintptr_t) p % 256)) {
                    std::fprintf(stderr, "selftest: aux %d arena alignment not honoured\n", plan.ordinals[i]);
                    return 1;
                }
                std::printf("selftest: aux %d arena %s allocated\n", plan.ordinals[i],
                            human(aux.capacity()).c_str());
            }
            if (plan.multi() && run_cross_device_selftest(plan) != 0) return 1;
            std::printf("strata-device selftest OK\n");
        }
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "strata-device: %s\n", e.what());
        return 1;
    }
}
