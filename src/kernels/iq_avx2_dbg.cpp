// src/kernels/iq_avx2_dbg.cpp - per-block isolation: which 256-value block of a row is wrong, type by type.
#include "strata/artifact/gguf_reader.hpp"
#include "strata/kernels/cpu/native_expert.hpp"
#include "strata/kernels/cpu/iq_avx2.hpp"
#include "ggml-cpu.h"
#include "ggml.h"

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace cpu = strata::kernels::cpu;
static const ggml_type_traits_cpu* traits(int type) { return ggml_get_type_traits_cpu((ggml_type) type); }

static float frand(unsigned& s) {
    s = s * 1664525u + 1013904223u;
    return (float) (s & 0xffffff) / 8388608.f - 1.f;
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: iq_avx2_dbg <shard1.gguf> layer [row]\n");
        return 2;
    }
    strata::GgufFile gguf(argv[1]);
    const int l = std::atoi(argv[2]);
    const int r0 = argc > 3 ? std::atoi(argv[3]) : 0;
    const int64_t H = 2560, FF = 640;
    const strata::TensorInfo* t[2] = {};
    const char* roles[2] = {"gate", "up"};
    for (const auto& ti : gguf.tensors())
        for (int r = 0; r < 2; ++r)
            if (ti.name == "blk." + std::to_string(l) + ".ffn_" + roles[r] + "_exps.weight") t[r] = &ti;
    if (!t[0]) {
        std::printf("no gate tensor\n");
        return 2;
    }
    // find the down tensor for the fmt
    const strata::TensorInfo* td = nullptr;
    for (const auto& ti : gguf.tensors())
        if (ti.name == "blk." + std::to_string(l) + ".ffn_down_exps.weight") td = &ti;
    cpu::NativeFmt f;
    std::string err;
    if (!cpu::native_fmt((int) t[0]->type, (int) td->type, H, FF, f, err)) {
        std::printf("fmt: %s\n", err.c_str());
        return 2;
    }
    const int nb = (int) f.n_embd / 256;
    std::printf("layer %d gu type %d, nb %d, gu_row %zu\n", l, f.gu_type, nb, f.gu_row);
    std::vector<uint8_t> row(f.gu_row);
    std::memcpy(row.data(), gguf.tensor_data(*t[0]) + (size_t) r0 * f.gu_row, f.gu_row);

    unsigned s = 12345u;
    std::vector<float> x(f.n_embd);
    for (auto& v : x) v = frand(s);
    std::vector<uint8_t> act(f.act_bytes);
    traits(f.gu_act)->from_float(x.data(), act.data(), f.n_embd);
    const ggml_vec_dot_t dot = traits(f.gu_type)->vec_dot;

    // full-row reference
    float full_ref = 0.f;
    dot((int) f.n_embd, &full_ref, 0, row.data(), 0, act.data(), 0, 1);
    float full_mine = 0.f;
    {
        float* op[1] = {&full_mine};
        const void* av[1] = {act.data()};
        cpu::iq256_rows(f.gu_type, row.data(), f.gu_row, (int) f.n_embd, av, 1, op, 0, 1);
    }
    std::printf("full row: ref %.9g mine %.9g %s\n", full_ref, full_mine, full_ref == full_mine ? "OK" : "DIFF");

    // per-block isolation: zero all other blocks, compare contributions
    const size_t bbytes = f.gu_row / (size_t) nb;
    for (int i = 0; i < nb; ++i) {
        std::vector<uint8_t> z(f.gu_row, 0);
        std::memcpy(z.data() + (size_t) i * bbytes, row.data() + (size_t) i * bbytes, bbytes);
        float br = 0.f, bm = 0.f;
        dot((int) f.n_embd, &br, 0, z.data(), 0, act.data(), 0, 1);
        float* op[1] = {&bm};
        const void* av[1] = {act.data()};
        cpu::iq256_rows(f.gu_type, z.data(), f.gu_row, (int) f.n_embd, av, 1, op, 0, 1);
        if (br != bm) std::printf("block %d: ref %.9g mine %.9g DIFF\n", i, br, bm);
    }
    std::printf("done\n");
    return 0;
}
