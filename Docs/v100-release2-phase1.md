# V100 Release 2 — Phase 1: the multi-device foundation (2026-10-02)

## 1. Mission

Release 2 goal (user-directed): **run the engine multi-GPU** on this box. The final target is
`cuda:0` + `cuda:1` — the two 32 GB V100s. For development **now**, the pair is `cuda:0` +
`cuda:2` (32 GB PCIe + 16 GB SXM2): GPU1's 32 GB hosts the resident `llama-server` Swift-27B
tenant (~32.2 GB), so it is effectively unusable until that tenant moves.

**Phase 1 is the foundation only**: the engine learns the machine's GPU set, establishes the
peer connections, and reports them — while the default single-device run stays **bit-identical**
to the frozen Release 1 baseline (proven by the Release 1 determinism anchor, §5). Phase 2 is
what actually moves state to the second card (§6).

## 2. Machine facts (measured 2026-10-02)

| | GPU0 | GPU1 | GPU2 | GPU3 | GPU4 |
|---|---|---|---|---|---|
| card | V100-PCIE-32GB | V100-PCIE-32GB | V100-SXM2-16GB | V100-SXM2-16GB | V100-SXM2-16GB |
| NUMA | 0 | 0 | 1 | 1 | 1 |
| connectivity to the others | PHB(1) | SYS(2,3,4) | PHB(3,4), SYS(0,1) | — | — |
| state today | strata.service | llama-server tenant (32.2 GB) | free | free | free |

- **No NVLink anywhere** — every inter-GPU path is PCIe (PHB/SYS).
- **P2P GPU0 ↔ GPU2: `none`** — `cudaDeviceCanAccessPeer` returns 0 in both directions at the
  driver level (the plan reports `p2p none`), so on the development pair the second card is only
  reachable through **host-staged transfers**. (GPU0 ↔ GPU1 — the final target pair, same NUMA
  node, PHB — cannot be probed until the tenant on GPU1 moves; the Phase 1 probe will say.)
- The 16 GB SXM2 cards sit on NUMA 1, the other NUMA node from GPU0's pool workers — the host
  staging buffer for Phase 2 should live on NUMA-allocated memory chosen per pair (noted for
  Phase 2; not implemented here).

## 3. What phase 1 implements

No engine numerics touched. Four code changes, one new header:

| file | change |
|---|---|
| `include/strata/core/devices.hpp` (new) | `DevicePlan`: ordinals (primary first), per-device `DeviceInfo`, the directed `p2p[i][j]` matrix (1 = peer access **enabled**, not merely possible), and `p2p_gbps` (filled only by the self-test probe). `make_device_plan(spec, probe)` and `device_plan_report(plan)`. |
| `src/core/device.cu` | the plan: spec parse/validate (bad or duplicate ordinal throws), per-device `device_info` (the sm_70 floor is enforced on **every** card, not just the primary), pairwise `cudaDeviceCanAccessPeer` + `cudaDeviceEnablePeerAccess`, and the timed `cudaMemcpyPeer` bandwidth probe (128 MiB blocks, ≥250 ms window, 1 GiB cap) used by the self-test only. |
| `src/program/generate.cpp` | `--devices A[,B]` (or `STRATA_DEVICES`, default `"0"`), resolved **before the first allocation**: the plan is built, the primary pinned with an explicit `cudaSetDevice`, and one startup report line goes to stderr (the service log). The plan stays alive for phase 2. |
| `src/core/device_main.cpp` | `strata-device` gains `--devices SPEC` and `--p2p`: it now builds the same plan the engine builds (same code path) and, with `--selftest`, verifies a 1 MiB pattern across the cards — through the peer connection when enabled, through host staging when not — plus an arena on every auxiliary device. |

**Ordinal semantics (important for the service):** `--devices` names ordinals in the
`CUDA_VISIBLE_DEVICES` namespace, i.e. post-remap. The production unit pins
`CUDA_VISIBLE_DEVICES=0` today, so a multi-GPU service must extend that environment variable
(development: `0,2`, engine spec `0,1`; final: `0,1` with both 32 GB cards, engine spec `0,1`
as well — same spec, different cards). The remap is the driver's, which is why the engine sees
mapped ordinals and why `strata-device --devices 0,2` (no remap) is the development spelling.

## 4. Design decisions

- **Peer enable at plan time, not first use.** A pointer into a second device's memory that is
  dereferenced before `cudaDeviceEnablePeerAccess` faults asynchronously; enabling before the
  first allocation (which this placement guarantees) makes the ordering uninteresting for every
  later phase.
- **Directed p2p matrix.** `p2p[i][j]` means "i can directly address j's memory". PCIe P2P is
  frequently asymmetric; a single bool would hide the case where one direction must stage
  through the host.
- **Bandwidth is measured, never assumed.** The placement decision of phase 2 (what lives where,
  how much to stage) is only as good as its transfer-cost model; `strata-device --p2p` measures
  the real D2D number per enabled pair. On the current development pair there is no enabled pair,
  so the honest answer is "none — stage through the host", which is what the report says.
- **Default is the Release 1 run.** `--devices 0` builds a one-device plan, pins device 0
  explicitly (a no-op where the engine already ran), and changes nothing else. The determinism
  gate below proves it, so phase 2 can build on a baseline that is provably the frozen one.

## 5. Gates (2026-10-02, GPU0, Release 1 build + phase 1 code)

| gate | result |
|---|---|
| clean build (sm_70, nvcc 12.9, C++20) | PASS — `strata` + `strata-device` link clean |
| `strata-device --devices 0,2` | PASS — both cards reported; `p2p none` (measured, §2) |
| `strata-device --devices 0,2 --p2p --selftest` | PASS — primary arena (64 MiB, poison, over-alloc refused), aux-2 arena, **cross-device 0→2 staged copy of 1 MiB pattern verified bit-for-bit** |
| `strata-device --selftest` (default, one card) | PASS — unchanged single-device behaviour |
| engine default devices, 32K/int8, g32 + det (fresh engine, wide ON) | PASS — g32 md5 `cf577e73…` (= Release 1 16K/32K g32 runs), **det md5 `cdb7f7d056f339ba704d3bb9620a1dec` = the Release 1 det anchor**, golden prefix MATCH (first token 271); startup now prints `strata devices: primary 0 (…31.43 GiB free…); p2p none` |
| engine with two devices visible (`CUDA_VISIBLE_DEVICES=0,2`, `--devices 0,1`), 32K/int8, g32 + det | PASS — report line shows primary 0 (32 GB) + aux 1 (16 GB SXM2), `p2p none`; **g32 and det byte-identical to the default-devices arm** (`s18check --diff`: byte-identical) |
| production service unaffected | PASS — `strata.service` restarted after the gates, `/health` `max_context 786432`, `images true` (786,432 + vision config from the post-Release-1 amendment) |

Raw data: `Logs/benchmarks/r2x-{g32,det1}-32768{,-dev2}.{json,log}`, `Logs/gpu/r2x-{g32,det1}-32768{,-dev2}.out|log`.

## 6. Phase 2 (next)

Move the context-scaling state to the auxiliary device and re-capture the layer graphs:

1. **What moves**: the QSA state — the int8 KV cache (k_q/v_q + scales), the block-pooled
   indexer state (idx_tail/idx_dead/idx_pooled), and the shared RoPE tables. At 786,432 that is
   the 9.3 GiB KV + ~1.9 GiB of QSA states of the session arena; the GDN recurrent states, MoE
   buffers, block buffers and PLE history (touched by every layer's kernels every token) stay
   on the primary.
2. **Transfer path (development pair)**: host staging — `p2p none` is measured, and the
   staging path is already exercised and verified by the phase 1 self-test. Per-token decode
   volume is bounded by QSA's selection: the attention reads at most `idx_top_k + idx_block - 1`
   = 2,051 cells, not the full context, so the per-token remote KV traffic is small; the indexer
   scan over the pooled state is the other stream and must be profiled.
3. **Graphs**: the layer graphs bake device pointers — they are re-captured with the split
   state; the capture path is already parameterised by `SessionState`, so this is a placement
   change, not a new graph.
4. **First to measure**: the P2P probe on the final 0,1 pair once the tenant on GPU1 moves —
   if the 32+32 pair has P2P, phase 2's hot path gets ~2× the staging bandwidth for free and the
   placement can shift toward keeping more state resident on the second card.
5. **Budget**: the 16 GB SXM2 holds ~14.5 GiB of state after overhead → 786,432 (9.3 GiB KV)
   fits with room for the indexer state; that is the first context step that a 32 GB single card
   cannot serve at full 8K expert cache.
