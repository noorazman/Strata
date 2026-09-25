# V100 VRAM Accounting — Strata MoE (Stage 1.1)

Purpose: a complete, measured VRAM ledger for the production config. Per the
Stage 1.1 brief, VRAM is **not** the optimization target — this document
exists so every later experiment can reason honestly about the budget, and so
the 16 GB card's fit is provable.

## GPU0 (32 GB PCIE), production config, 8192 ctx fp16 KV

| component | size | source |
|---|---:|---|
| dense weights from pack (`dense.bin`) | 1,466 MiB | startup: "1466 MiB of weights loaded" |
| 300 native projection matrices (GGUF shard 1) | 1,781 MiB | startup: "300 native projection matrices, 1781.11 MiB" |
| MTP draft layer + draft head | 876 MiB | "808 MiB of VRAM (experts 675, dense 111)" + "68.0 MiB" head |
| LM head (native Q5_K) | 417 MiB | "experimental native Q5_K head, 437043200 bytes" |
| verify-window device buffers | 57.8 MiB | "window up to 4 tokens, 57.8 MiB" |
| expert cache, 8,000 slots (auto, profile-filled) | 12,930 MiB (12.93 GiB) | "expert cache auto: 26.88 GiB free, 700 MiB reserved -> 8000 slots" |
| KV cache, 8192 ctx fp16 | ~135 MiB | measured below |
| **subtotal** | **17,663 MiB (17.25 GiB)** | |
| CUDA context + cuBLAS workspaces + graph scratch + alignment | ~1,230 MiB | residual |
| **measured peak (nvidia-smi, 10 Hz poll)** | **18,896 MiB (18.45 GiB)** | phase0 256-token run |

Token embeddings (260 MiB) live in **mapped host memory**, not VRAM
("token embedding type 21 in mapped host memory").

## KV cache, measured by context-size delta

| max-context | peak VRAM | Δ vs 2048 |
|---:|---:|---:|
| 2,048 | 18,706 MiB | — |
| 4,096 | 18,762 MiB | +56 MiB |
| 8,192 | 18,896 MiB | +190 MiB |

≈ 27–32 KB per context token → the 8192 KV cache is ~130 MiB, i.e. **<1 % of
the footprint**. KV size is not a lever on this card (and `--kv int8` is
slower anyway — see `v100-performance.md`).

## GPU4 (16 GB SXM2), same build

- "expert cache auto: 10.91 GiB free, 700 MiB reserved -> 4715 slots", then
  the profile/VRAM fit settles on **6,321 slots / 10.23 GiB** (identical to
  Stage 1's fit).
- Measured peak: **16,133 MiB** (15.76 GiB) over 32- and 256-token runs —
  271 MiB (1.6 %) of headroom. Stage 1 measured 15,949 MiB with its
  sampler; the 184 MiB delta is sampling-phase, both fit comfortably.
- The 2,047-token prompt path borrows 923 cache slots (1.50 GiB) from the
  expert cache for prompt buffers ("prompt path borrows 923 cache slots"),
  refilled in 173.6 ms after the prefill.

## Consequences for Stage 1.1 experiments

- `--vram-reserve-mib 300` (vs 700): auto cache still 8,000 slots — with a
  profile, **the cache is capped at the profile's 8,000 ranked pairs**, not
  by free VRAM (free-based size would be ~8,300+).
- `--expert-cache 12000/16000`: fits fine, but under the PROFILE policy the
  extra slots stay unfilled (0 admitted) — hit pattern unchanged (see
  `v100-performance.md`).
- 16 GB fit is stable across all tested knob variants (peaks 16,111–16,133
  MiB); nothing in Stage 1.1 grew the footprint.
