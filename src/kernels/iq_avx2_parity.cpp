// src/kernels/iq_avx2_parity.cpp - Stage 1.4: bit-exact parity of iq_avx2 vs the per-token ggml-cpu loop.
//
//     build-sm70/iq_avx2_parity <shard1.gguf> [layer ...]
//
// For each layer (or a subset) this builds one expert's gate/up/down rows from the GGUF, quantizes
// deterministic test activations the way the pool does, and compares the AVX2 decode-once rows against
// the per-token ggml-cpu dot loop element by element.  Any mismatch is printed with (type, nt, row, token).
#include "strata/artifact/gguf_reader.hpp"
#include "strata/kernels/cpu/native_expert.hpp"
#include "strata/kernels/cpu/iq_avx2.hpp"
#include "ggml-cpu.h"
#include "ggml.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace cpu = strata::kernels::cpu;

static const ggml_type_traits_cpu* traits(int type) { return ggml_get_type_traits_cpu((ggml_type) type); }

// deterministic pseudo-random float in [-1, 1)
static float frand(unsigned& s) {
    s = s * 1664525u + 1013904223u;
    return (float) (s & 0xffffff) / 8388608.f - 1.f;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: iq_avx2_parity <shard1.gguf> [layer ...]\n");
        return 2;
    }
    strata::GgufFile gguf(argv[1]);
    std::vector<int> layers;
    for (int i = 2; i < argc; ++i) layers.push_back(std::atoi(argv[i]));
    if (layers.empty())
        for (int l = 0; l < 48; ++l) layers.push_back(l);
    const int NTMAX = 4;
    const int64_t H = 2560, FF = 640;
    const int E = 0;  // expert
    int mismatches = 0, checked = 0;

    for (int l : layers) {
        const strata::TensorInfo* t[3] = {};
        const char* roles[3] = {"gate", "up", "down"};
        for (const auto& ti : gguf.tensors())
            for (int r = 0; r < 3; ++r)
                if (ti.name == "blk." + std::to_string(l) + ".ffn_" + roles[r] + "_exps.weight") t[r] = &ti;
        if (!t[0] || !t[1] || !t[2]) {
            std::printf("layer %d: no expert tensors\n", l);
            continue;
        }
        cpu::NativeFmt f;
        std::string err;
        if (!cpu::native_fmt((int) t[0]->type, (int) t[2]->type, H, FF, f, err)) {
            std::printf("layer %d: %s\n", l, err.c_str());
            continue;
        }
        std::vector<uint8_t> blob(f.bytes);
        std::memcpy(blob.data(), gguf.tensor_data(*t[0]) + (size_t) E * f.up_off, f.up_off);
        std::memcpy(blob.data() + f.up_off, gguf.tensor_data(*t[1]) + (size_t) E * f.up_off, f.up_off);
        std::memcpy(blob.data() + f.down_off, gguf.tensor_data(*t[2]) + (size_t) E * f.d_row * (size_t) H,
                    f.d_row * (size_t) H);
        const bool gu_ok = cpu::iq256_supported(f.gu_type);
        const bool dn_ok = cpu::iq256_supported(f.d_type);
        if (!gu_ok && !dn_ok) {
            std::printf("layer %d: gu type %d, down type %d - not an iq_avx2 pair, skipped\n", l, f.gu_type, f.d_type);
            continue;
        }

        // test activations, quantized the pool way
        std::vector<std::vector<uint8_t>> act(NTMAX + 1), hq(NTMAX + 1);
        std::vector<std::vector<const void*>> av(NTMAX + 1), hv(NTMAX + 1);
        for (int nt = 1; nt <= NTMAX; ++nt) {
            act[nt].resize((size_t) nt * f.act_bytes);
            hq[nt].resize((size_t) nt * f.h_bytes);
            av[nt].resize(nt);
            hv[nt].resize(nt);
            for (int tt = 0; tt < nt; ++tt) {
                unsigned s = (unsigned) (l * 1000 + tt);
                std::vector<float> x(f.n_embd), h(f.n_ff);
                for (int i = 0; i < (int) f.n_embd; ++i) x[i] = frand(s);
                traits(f.gu_act)->from_float(x.data(), act[nt].data() + (size_t) tt * f.act_bytes, f.n_embd);
                for (int i = 0; i < (int) f.n_ff; ++i) h[i] = frand(s);
                traits(f.d_act)->from_float(h.data(), hq[nt].data() + (size_t) tt * f.h_bytes, f.n_ff);
            }
            for (int tt = 0; tt < nt; ++tt) {
                av[nt][tt] = act[nt].data() + (size_t) tt * f.act_bytes;
                hv[nt][tt] = hq[nt].data() + (size_t) tt * f.h_bytes;
            }
        }

        const ggml_vec_dot_t dotg = traits(f.gu_type)->vec_dot;
        const ggml_vec_dot_t dotd = traits(f.d_type)->vec_dot;
        const int n = (int) f.n_embd;

        for (int nt = 1; nt <= NTMAX; ++nt) {
            // ---- gate/up ----
            std::vector<std::vector<float>> ref(NTMAX, std::vector<float>(FF)), out(NTMAX, std::vector<float>(FF));
            for (int r = 0; r < (int) FF; ++r) {
                const uint8_t* gr = blob.data() + (size_t) r * f.gu_row;
                const uint8_t* ur = blob.data() + f.up_off + (size_t) r * f.gu_row;
                for (int tt = 0; tt < nt; ++tt) {
                    float g = 0.f, u = 0.f;
                    dotg(n, &g, 0, gr, 0, av[nt][tt], 0, 1);
                    dotg(n, &u, 0, ur, 0, av[nt][tt], 0, 1);
                    ref[tt][r] = (g / (1.f + std::exp(-g))) * u;
                }
            }
            if (gu_ok) {
                float* outptr[NTMAX];
                for (int t = 0; t < NTMAX; ++t) outptr[t] = out[t].data();
                cpu::iq256_gu_rows(f.gu_type, blob.data(), f.gu_row, f.up_off, n, av[nt].data(), nt, outptr, 0,
                                    (int) FF);
                for (int r = 0; r < (int) FF; ++r)
                    for (int tt = 0; tt < nt; ++tt)
                        if (out[tt][r] != ref[tt][r]) {
                            std::printf("MISMATCH gu l%d type %d nt %d r %d t %d: mine %.9g ref %.9g\n", l, f.gu_type,
                                        nt, r, tt, out[tt][r], ref[tt][r]);
                            if (++mismatches >= 40) {
                                std::printf("too many mismatches, stopping\n");
                                return 1;
                            }
                        }
                checked += (int) FF;
            }
            // ---- down ----
            std::vector<std::vector<float>> dref(NTMAX, std::vector<float>(H)), dout(NTMAX, std::vector<float>(H));
            for (int r = 0; r < (int) H; ++r) {
                const uint8_t* dr = blob.data() + f.down_off + (size_t) r * f.d_row;
                for (int tt = 0; tt < nt; ++tt) {
                    float s = 0.f;
                    dotd((int) f.n_ff, &s, 0, dr, 0, hv[nt][tt], 0, 1);
                    dref[tt][r] = s;
                }
            }
            if (dn_ok) {
                float* dnptr[NTMAX];
                for (int t = 0; t < NTMAX; ++t) dnptr[t] = dout[t].data();
                cpu::iq256_rows(f.d_type, blob.data() + f.down_off, f.d_row, (int) f.n_ff, hv[nt].data(), nt, dnptr,
                                 0, (int) H);
                for (int r = 0; r < (int) H; ++r)
                    for (int tt = 0; tt < nt; ++tt)
                        if (dout[tt][r] != dref[tt][r]) {
                            std::printf("MISMATCH dn l%d type %d nt %d r %d t %d: mine %.9g ref %.9g\n", l, f.d_type,
                                        nt, r, tt, dout[tt][r], dref[tt][r]);
                            if (++mismatches >= 40) {
                                std::printf("too many mismatches, stopping\n");
                                return 1;
                            }
                        }
                checked += (int) H;
            }
        }
        std::printf("layer %d: gu type %d %s, down type %d %s, %d rows x nt<=4 checked so far\n", l, f.gu_type,
                    gu_ok ? "ok" : "skip", f.d_type, dn_ok ? "ok" : "skip", checked);
    }
    std::printf(mismatches ? "PARITY FAILED: %d mismatches\n" : "PARITY OK: bit-exact across %d layer rows\n", mismatches);
    return mismatches ? 1 : 0;
}
