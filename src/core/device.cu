// src/core/device.cu - P2.S1: the CUDA side of the runtime core.
#include "strata/core/device.hpp"
#include "strata/core/devices.hpp"

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace strata::core {

namespace {

void check(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        throw CudaError(std::string(what) + ": " + cudaGetErrorString(e), (int) e);
    }
}

// A NaN pattern, not zero.  Zeros read from uninitialised memory are indistinguishable from real zeros in a
// dequantized weight or a masked attention score, which is exactly the kind of wrong-but-plausible value the
// Phase 1 harnesses kept catching.
__global__ void poison_kernel(float* p, uint64_t n_floats) {
    const uint64_t i = (uint64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n_floats) p[i] = __int_as_float(0x7fc00000);
}

#if defined(STRATA_USE_HIP)
#if !defined(STRATA_HIP_ARCHS)
#error "STRATA_HIP_ARCHS (the compiled HIP architectures) is set by cmake/hip_backend.cmake"
#endif
// "gfx1201:sramecc-:xnack-" -> "gfx1201"
std::string base_arch(const char* gcn_arch_name) {
    std::string arch(gcn_arch_name);
    const size_t colon = arch.find(':');
    if (colon != std::string::npos) arch.resize(colon);
    return arch;
}

bool compiled_for(const std::string& arch) {
    const std::string list = STRATA_HIP_ARCHS;
    size_t a = 0;
    while (a <= list.size()) {
        size_t b = list.find(',', a);
        if (b == std::string::npos) b = list.size();
        if (!arch.empty() && list.compare(a, b - a, arch) == 0 && b - a == arch.size()) return true;
        a = b + 1;
    }
    return false;
}

std::string arch_problem(const cudaDeviceProp& p, int ordinal) {
    const std::string arch = base_arch(p.gcnArchName);
    const std::string card = "GPU " + std::to_string(ordinal) + " (" + p.name + ", " + arch + ")";
    if (!compiled_for(arch)) {
        return card + " is not an architecture this Strata engine was compiled for (" + STRATA_HIP_ARCHS +
               "); compile it for this card (./setup.sh --backend hip, or -DCMAKE_HIP_ARCHITECTURES=" + arch +
               ", docs/AMD_HIP.md) or choose another GPU with HIP_VISIBLE_DEVICES";
    }
    if (p.warpSize != 32) {
        return card + " runs wave" + std::to_string(p.warpSize) + "; Strata's HIP kernels need wave32";
    }
    return "";
}
#endif

}  // namespace

const char* compiled_gpu_archs() {
#if defined(STRATA_USE_HIP)
    return STRATA_HIP_ARCHS;
#else
    return "";
#endif
}

int device_count() {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess) {   // HIP without a usable device reports an error, not 0
        cudaGetLastError();
        return 0;
    }
    return count < 0 ? 0 : count;
}

bool device_summary(int ordinal, std::string& name, std::string& detail) {
    cudaDeviceProp p{};
    if (ordinal < 0 || ordinal >= device_count() || cudaGetDeviceProperties(&p, ordinal) != cudaSuccess) {
        cudaGetLastError();
        return false;
    }
    char buf[160];
#if defined(STRATA_USE_HIP)
    std::snprintf(buf, sizeof(buf), "arch %s, %.1f GiB, wave%d", base_arch(p.gcnArchName).c_str(),
                  (double) p.totalGlobalMem / (1024.0 * 1024 * 1024), p.warpSize);
#else
    std::snprintf(buf, sizeof(buf), "compute capability %d.%d, %.1f GiB", p.major, p.minor,
                  (double) p.totalGlobalMem / (1024.0 * 1024 * 1024));
#endif
    name = p.name;
    detail = buf;
    return true;
}

std::string gpu_arch_problem(int ordinal) {
#if defined(STRATA_USE_HIP)
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || ordinal < 0 || ordinal >= count) {
        cudaGetLastError();
        return "";
    }
    cudaDeviceProp p{};
    if (cudaGetDeviceProperties(&p, ordinal) != cudaSuccess) {
        cudaGetLastError();
        return "";
    }
    return arch_problem(p, ordinal);
#else
    (void) ordinal;
    return "";
#endif
}

std::string device_code_error() {
#if defined(STRATA_USE_HIP)
    return "";   // gpu_arch_problem() checks the HIP architectures against STRATA_HIP_ARCHS, before this point
#else
    // every .cu of the engine is compiled for the same CMAKE_CUDA_ARCHITECTURES, so this kernel stands for all
    cudaFuncAttributes a{};
    const cudaError_t e = cudaFuncGetAttributes(&a, poison_kernel);
    if (e == cudaSuccess) return {};
    cudaGetLastError();
    return cudaGetErrorString(e);
#endif
}

DeviceInfo device_info(int ordinal) {
    int count = 0;
    check(cudaGetDeviceCount(&count), "cudaGetDeviceCount");
    if (count == 0) {
#if defined(STRATA_USE_HIP)
        throw CudaError(std::string("no HIP device is present; this engine was compiled for ") + STRATA_HIP_ARCHS, -1);
#else
        throw CudaError("no CUDA device is present; Strata needs sm_70 (Volta) or newer, "
                           "developed on sm_120 (RTX 5000 series)", -1);
#endif
    }
    if (ordinal < 0 || ordinal >= count) {
        throw CudaError("device ordinal " + std::to_string(ordinal) + " is out of range (have " +
                            std::to_string(count) + ")",
                        -1);
    }
    DeviceInfo d;
    d.ordinal = ordinal;
    check(cudaSetDevice(ordinal), "cudaSetDevice");

    cudaDeviceProp p{};
    check(cudaGetDeviceProperties(&p, ordinal), "cudaGetDeviceProperties");
    d.name = p.name;
    d.cc_major = p.major;
    d.cc_minor = p.minor;
    d.multi_processor_count = p.multiProcessorCount;

    size_t free_b = 0, total_b = 0;
    check(cudaMemGetInfo(&free_b, &total_b), "cudaMemGetInfo");
    d.free_bytes = free_b;
    d.total_bytes = total_b;

    check(cudaDriverGetVersion(&d.driver_version), "cudaDriverGetVersion");
    check(cudaRuntimeGetVersion(&d.runtime_version), "cudaRuntimeGetVersion");

    // The engine is developed and measured against sm_120.  CMake enforces the sm_70 floor at COMPILE time;
    // RUNNING on something older is caught here, because a binary can be carried to a machine with an older
    // card and would otherwise silently take whatever path the driver chose.  The HIP
    // backend checks the card against the architectures the binary was compiled for (and wave32).
#if defined(STRATA_USE_HIP)
    d.arch = base_arch(p.gcnArchName);
    if (const std::string why = arch_problem(p, ordinal); !why.empty()) throw CudaError(why, -1);
#else
    // #236: the experimental build (-DSTRATA_EXPERIMENTAL_SM60=ON: Pascal sm_60) runs on the cards it
    // was built for; the release engine keeps the sm_70 (Volta, e.g. Tesla V100) floor.
#if defined(STRATA_EXPERIMENTAL_SM60)
    constexpr int kMinCc = 60;
    const char* const kNeed = "sm_60 (Pascal) or newer - this is the experimental Pascal build";
#else
    constexpr int kMinCc = 70;
    const char* const kNeed = "sm_70 (Volta, e.g. Tesla V100) or newer - sm_120 "
                              "(RTX 5000 series / Blackwell) is the reference target";
#endif
    if (d.cc_major * 10 + d.cc_minor < kMinCc) {
        throw CudaError("device " + d.name + " reports compute capability " + std::to_string(d.cc_major) +
                            "." + std::to_string(d.cc_minor) + "; Strata needs compute capability " + kNeed,
                        -1);
    }
#endif
    return d;
}

DeviceArena::DeviceArena(uint64_t bytes, int ordinal, bool poison)
    : capacity_(bytes), ordinal_(ordinal), poison_(poison) {
    if (bytes == 0) throw CudaError("DeviceArena of 0 bytes", -1);
    check(cudaSetDevice(ordinal), "cudaSetDevice");
    // One allocation for the whole region.  cudaMalloc of a large block is the thing that can fail late, so it
    // happens once, here, before anything depends on it.
    check(cudaMalloc(&base_, (size_t) bytes), "cudaMalloc");
    if (poison_) {
        const int threads = 256;
        const uint64_t n = bytes / sizeof(float);
        const uint64_t blocks = (n + threads - 1) / threads;
        // gridDim.x is 32-bit, so a large region needs a loop.  12 GB of floats is 3e9 elements = 1.2e7
        // blocks, which fits, but the loop keeps it correct for any size rather than for today's sizes.
        const uint64_t max_blocks = 0x7FFFFFFFull;
        for (uint64_t b = 0; b < blocks; b += max_blocks) {
            const uint64_t chunk = (blocks - b < max_blocks) ? (blocks - b) : max_blocks;
            poison_kernel<<<(unsigned) chunk, threads>>>((float*) base_ + b * threads, n - b * threads);
            check(cudaGetLastError(), "poison_kernel");
        }
        check(cudaDeviceSynchronize(), "poison sync");
    }
}

DeviceArena::~DeviceArena() {
    if (base_) cudaFree(base_);          // best effort: a destructor must not throw
}

void* DeviceArena::alloc(uint64_t bytes, uint64_t align) {
    if (bytes == 0) return nullptr;
    if (align == 0 || (align & (align - 1)) != 0) {
        throw CudaError("DeviceArena::alloc alignment must be a power of two", -1);
    }
    const uint64_t start = (used_ + align - 1) & ~(align - 1);
    if (start + bytes > capacity_) {
        char msg[256];
        std::snprintf(msg, sizeof(msg),
                      "DeviceArena out of memory: asked for %llu B at offset %llu (align %llu) in a %llu B "
                      "region - the plan from P1.S9 did not close",
                      (unsigned long long) bytes, (unsigned long long) start, (unsigned long long) align,
                      (unsigned long long) capacity_);
        throw CudaError(msg, -1);
    }
    used_ = start + bytes;
    return (char*) base_ + start;
}

// ================================ Release 2, phase 1: the multi-device plan ================================

int DevicePlan::index_of(int ordinal) const {
    for (size_t i = 0; i < ordinals.size(); ++i)
        if (ordinals[i] == ordinal) return (int) i;
    return -1;
}

namespace {

std::vector<int> parse_device_spec(const std::string& spec) {
    std::vector<int> out;
    size_t start = 0;
    while (start <= spec.size()) {
        const size_t comma = spec.find(',', start);
        std::string tok = (comma == std::string::npos) ? spec.substr(start) : spec.substr(start, comma - start);
        const size_t a = tok.find_first_not_of(" \t"), b = tok.find_last_not_of(" \t");
        if (a == std::string::npos) throw CudaError("empty device ordinal in spec '" + spec + "'", -1);
        tok = tok.substr(a, b - a + 1);
        char* endp = nullptr;
        const long v = std::strtol(tok.c_str(), &endp, 10);
        if (endp == tok.c_str() || *endp != '\0' || v < 0 || v > 127) {
            throw CudaError("bad device ordinal '" + tok + "' in spec '" + spec + "'", -1);
        }
        out.push_back((int) v);
        if (comma == std::string::npos) break;
        start = comma + 1;
    }
    return out;
}

/// The measured device-to-device bandwidth of one ENABLED pair, in GB/s: 128 MiB blocks of `cudaMemcpyPeer`
/// from src to dst, repeated until at least 250 ms of transfers have run (a single copy is too short to
/// average out the PCIe arbitration noise on this box) and capped at 1 GiB of measured traffic.  The events
/// live on the SOURCE device: `cudaMemcpyPeer` runs from the source's context, and every iteration
/// synchronises the stop event, so after the loop `a`/`b` span exactly the measured copies.
double measure_p2p_gbps(int src, int dst) {
    // The ordinals are MEMBERS, not references to the enclosing function's parameters: in a .cu file a
    // non-trivial destructor is compiled for device use as well, where enclosing-function locals are
    // out of reach, and the reference is exactly the error that costs the build.
    struct Guard {
        int src = -1, dst = -1;
        void* s = nullptr;
        void* d = nullptr;
        cudaEvent_t a = nullptr, b = nullptr;
        ~Guard() {
            if (a) cudaEventDestroy(a);
            if (b) cudaEventDestroy(b);
            if (d) { cudaSetDevice(dst); cudaFree(d); }
            if (s) { cudaSetDevice(src); cudaFree(s); }
        }
    } g;
    g.src = src;
    g.dst = dst;
    const uint64_t bytes = 128ull << 20;
    check(cudaSetDevice(src), "cudaSetDevice(src)");
    check(cudaMalloc(&g.s, (size_t) bytes), "p2p probe: cudaMalloc on the source");
    check(cudaSetDevice(dst), "cudaSetDevice(dst)");
    check(cudaMalloc(&g.d, (size_t) bytes), "p2p probe: cudaMalloc on the destination");
    // Seed the source with a pattern, not zeros: a pattern makes a transfer that silently moves nothing
    // visible if this probe is ever run by hand.
    check(cudaMemset(g.s, 0x5a, (size_t) bytes), "p2p probe: seed");
    check(cudaSetDevice(src), "cudaSetDevice(src) again");
    check(cudaEventCreate(&g.a), "p2p probe: event a");
    check(cudaEventCreate(&g.b), "p2p probe: event b");
    check(cudaMemcpyPeer(g.d, dst, g.s, src, (size_t) bytes), "p2p probe: warmup");
    check(cudaEventRecord(g.a, 0), "p2p probe: event a record");
    uint64_t measured = 0;
    for (;;) {
        check(cudaMemcpyPeer(g.d, dst, g.s, src, (size_t) bytes), "p2p probe: copy");
        measured += bytes;
        check(cudaEventRecord(g.b, 0), "p2p probe: event b record");
        check(cudaEventSynchronize(g.b), "p2p probe: event b sync");
        float ms = 0;
        check(cudaEventElapsedTime(&ms, g.a, g.b), "p2p probe: elapsed");
        if (ms >= 250.0f || measured >= (uint64_t) (1 << 30)) break;
    }
    float ms = 0;
    check(cudaEventElapsedTime(&ms, g.a, g.b), "p2p probe: elapsed final");
    return (double) measured / (ms / 1000.0) / 1e9;
}

}  // namespace

DevicePlan make_device_plan(const std::string& spec, bool probe_bandwidth) {
    const std::vector<int> ordinals = parse_device_spec(spec);
    if (ordinals.empty()) throw CudaError("empty device spec", -1);
    for (size_t i = 0; i < ordinals.size(); ++i)
        for (size_t j = i + 1; j < ordinals.size(); ++j)
            if (ordinals[i] == ordinals[j]) {
                throw CudaError("duplicate ordinal " + std::to_string(ordinals[i]) + " in spec '" + spec + "'", -1);
            }

    DevicePlan plan;
    plan.ordinals = ordinals;
    for (int o : ordinals) plan.info.push_back(device_info(o));      // throws on a card below sm_70
    const size_t n = ordinals.size();
    plan.p2p.assign(n, std::vector<int>(n, 0));
    plan.p2p_gbps.assign(n, std::vector<double>(n, 0.0));
    for (size_t i = 0; i < n; ++i) plan.p2p[i][i] = 1;
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            if (i == j) continue;
            int can = 0;
            check(cudaDeviceCanAccessPeer(&can, ordinals[i], ordinals[j]), "cudaDeviceCanAccessPeer");
            if (!can) continue;
            // Enable from i's context, addressing j: i's kernels may then dereference j's pointers directly.
            check(cudaSetDevice(ordinals[i]), "cudaSetDevice for the peer enable");
            const cudaError_t e = cudaDeviceEnablePeerAccess(ordinals[j], 0);
            if (e != cudaSuccess && e != cudaErrorAlreadyAcquired) {
                throw CudaError("cudaDeviceEnablePeerAccess: device " + std::to_string(ordinals[i]) +
                                    " cannot address device " + std::to_string(ordinals[j]) + ": " +
                                    cudaGetErrorString(e),
                                (int) e);
            }
            plan.p2p[i][j] = 1;
            if (probe_bandwidth) plan.p2p_gbps[i][j] = measure_p2p_gbps(ordinals[i], ordinals[j]);
        }
    }
    return plan;
}

std::string device_plan_report(const DevicePlan& plan) {
    auto human = [](uint64_t b) {
        char buf[32];
        std::snprintf(buf, sizeof(buf), "%.2f GiB", (double) b / (1024.0 * 1024 * 1024));
        return std::string(buf);
    };
    std::string s = "strata devices: ";
    for (size_t i = 0; i < plan.ordinals.size(); ++i) {
        const DeviceInfo& d = plan.info[i];
        if (i) s += "; ";
        s += (i == 0 ? "primary " : "aux ") + std::to_string(d.ordinal) + " (" + d.name + ", sm_" +
             std::to_string(d.cc_major) + std::to_string(d.cc_minor) + ", " + human(d.free_bytes) + " free / " +
             human(d.total_bytes) + " total)";
    }
    s += "; p2p ";
    bool any = false;
    for (size_t i = 0; i < plan.p2p.size(); ++i)
        for (size_t j = 0; j < plan.p2p.size(); ++j) {
            if (i == j || !plan.p2p[i][j]) continue;
            if (any) s += ", ";
            s += std::to_string(plan.ordinals[i]) + "->" + std::to_string(plan.ordinals[j]) +
                 (plan.p2p_gbps[i][j] > 0 ? " (" + std::to_string((long long) (plan.p2p_gbps[i][j] * 100 + 0.5)) +
                                                " MB/s)"
                                          : std::string());
            any = true;
        }
    if (!any) s += "none";
    return s;
}

}  // namespace strata::core
