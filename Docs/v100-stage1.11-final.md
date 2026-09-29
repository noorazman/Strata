# V100 Stage 1.11 — Prefill / TTFT Optimization (profile → isolate → measure)

Production target: 32 GB Tesla V100 (GPU0 only). Started from the clean Stage 1.10 commit
`25fbc76` (branch `stage1.3-expert-pool-sync`). Goal: break the ~2.17 ms/token prefill cost into
categories, identify the largest *real* bottleneck, then make ONE isolated opt-in change at a time
→ benchmark → correctness-gate, keeping all changes opt-in and the 32K production config untouched.

**Bottom line: the prefill wall is a saturated SM + DMA pipeline (~89 % GPU busy, MoE expert
streaming ~55 % of wall). Three isolated candidates were measured — E3 (MTP draft/prefill
overlap), E4 (async routing-ids host round-trip), E7 (two-stream staging DMA). All three, alone
and combined, are net-neutral on TTFT (≤ 0.5 s on a 40 s wall at 16 K, ≤ 0.3 % at 32 K), each
correctness-clean (32/32 golden, 256/256 byte-identical, cold-request output bit-identical to the
baseline). Per the stage rule ("if no optimization produces a meaningful improvement, document why
and stop"): no production default changed; all three stay opt-in (`STRATA_MTP_OVERLAP`,
`STRATA_MOE_ASYNC_IDS`, `STRATA_TWO_STREAM_DMA`, all default OFF); the 32K production config and
`strata.service` were restored and verified. The why is in §6.**

---

## 1. Method

- **Profiling**: nsys 2025.1.3 serve-mode capture of a 16 K int8 prefill (production flags
  `--prefill 2048 --spec 4 --spec-min-p 0.5 --mtp mtp/rt`, `STRATA_TTFT=1`), parsed with
  `bench/v100/s111_breakdown.py` into a 10-category kernel-time breakdown; DMA gap analysis
  directly on the `CUPTI_ACTIVITY_KIND_MEMCPY` table (per-stream active throughput + inter-transfer
  gap distribution, correlated with concurrent kernel activity). ncu 2025.2.1 SOL campaign
  (`bench/v100/s111ncu2.sh`, 11 kernel profiles, application-replay, fixed `--expert-cache 9000` so
  replay passes see identical launch counts, `wait_idle()` between profiles) — results in §2.3.
- **A/B**: `bench/v100/s110.sh ctx <C>` (serve path, per-context config, 1 K warmup + two
  fill-prompt runs at C−overhead−264 tokens, 256 max-new, VRAM CSV) with `EXTRA_ENV` for the
  opt-in flags; same-build OFF and ON arms at 4 K/16 K/32 K. Client-side `t_first_reasoning_ms`
  (= TTFT) compared per run; r1 = first (cold-cache) request, r2 = second (warm) request.
- **Correctness gates** (per experiment, fresh engine each): `s111e3/e4/e7.sh gate <C>` =
  32-token golden-prefix (`s18check.py --prefix`, GOLDEN_32) + 256-token ×2 byte-identical
  determinism (int8 KV, production flags, `--max-context C`), decode-rate check (target ~49–50
  tok/s). Shared-engine r1/r2 fill runs are the *pre-existing* GPU-hit-path caveat (documented in
  Stage 1.10 — warm expert-cache residency changes the hit path); the cold r1 output is the
  bit-identical check and matched the OFF baseline in every arm.
- **Environment**: `CUDA_VISIBLE_DEVICES=0`, `ulimit -l unlimited`, `strata.service` stopped for
  all measurement; `ninfer-serve` (port 8001) untouched; GPU0 idle-checked (≤ 2 GiB) before every
  run.

## 2. Profiling — the prefill breakdown (16 K, nsys)

### 2.1 Ten-category kernel-time breakdown (16 K prefill, 36.57 s wall = 2.232 ms/tok)

| category | ms | % of wall | notes |
|---|---|---|---|
| MoE section (stage+dequant+GEMM+combine) | 20,072 | 54.9 % | the long pole; DMA-rate-bound (§2.2) |
| dequant (all) | 9,442 | 25.8 % | dequant_gu dominant; mostly inside MoE |
| expert GEMM (magma) | 4,686 | 12.8 % | inside MoE |
| other kernels | 16,706 | 45.7 % | magma-dense 4,439, Kernel2 3,492, dequant_flat 3,126, QSA attn_chunk 2,674 (7.3 %), gdn_rec 1,387, bf16_mmvf 1,377, volta 1,148, splitK 967, dequant_kernel 803, swiglu 761 (categories overlap — MoE spans several) |
| host-side sync | 1,008 | 2.8 % | 384 × 80 KB **pageable** D2H of routing ids + full `cudaStreamSynchronize(m.cs)` per MoE layer (§4, E4) |
| MTP draft (K/V build at chunk boundaries) | ~1,370 | 3.7 % | 8 boundaries × ~171 ms serialized, host-synced sub-chunks (§5, E3) |

Bottleneck verdict (carried from Stage 1.10, now refined): **MoE expert weight streaming —
171.5 GB H2D during the 16 K prefill at 10.57 GB/s ≈ 80 % of the PCIe3 x16 ceiling — plus dequant
(25.8 %). The SMs are ~89 % busy; the wall is the fused SM+DMA pipeline, not any single gap.**

### 2.2 The DMA picture (what the trace actually shows)

From the memcpy table (stream 15 = the prefill staging ring, 96,650 × ~2 MB expert blobs = 169.7 GB):

- **While active: 10.57 GB/s** — the copy engine itself only achieves ~80 % of the link even when
  continuously fed (per-CE TLP/MDR efficiency at ~2 MB transfers; a second CE measured the same
  ~10 GB/s active in the same trace).
- **19.9 s of CE idle gaps** in the 124 s capture, in two distinct shapes:
  - **8 × ~270 ms gaps** — one per 2048-token chunk boundary. During each gap the **SMs are busy
    (18,540 kernels, 249 ms of kernel time in-window)** — that is the MTP draft K/V build (the E3
    target). The CE is idle because the next chunk's routing (and thus its expert set) is unknown
    until the chunk's GDN/attention/route work finishes.
  - **371 gaps of 1–100 ms (mean 36 ms, 13.4 s total)** — one per MoE *section* (layer): the CE
    starves while the host does the per-layer routing round-trip (route GEMM → 80 KB ids D2H →
    full-stream sync → host count-sort) plus the layer's non-MoE prefix (GDN/attention/GEMM).
  - **Within a section: no gaps > 1 ms** — the 8-slot ring with 7-deep lookahead keeps the single
    CE fully fed (this is why E7's expected gain was only the active-rate term, §7).

### 2.3 ncu SOL campaign (per-kernel roofline)

Ran after the A/B matrix (same build, 4 K-token prefill, `--expert-cache 9000` fixed so all
replay passes see identical launch counts). ncu 2025.2.1, SpeedOfLight section,
application-replay mode (each profiled kernel is re-measured across 16 full app passes,
~30–35 min per profile), `--kill 1` so the app never reaches the ncu-fragile MTP decode.
Per launch: duration plus % of peak for SM (compute), memory, DRAM and L2.

**ncu `-k` filter gotcha** — the filter matches the *function* name base, not the nsys-style
demangled name: the cutlass GEMMs nsys shows as `cutlass::Kernel2<cutlass_70_wmma_…>` are just
**`Kernel2`** in ncu (all tile variants share the name), the other cuBLAS GEMMs are
`volta_s884gemm_fp16_*`, and the split-K reduces `splitKreduce_kernel` (no `cublasLt::` prefix).
When a filter matches nothing, ncu prints the complete observed-kernel list at the end of the
run log — the campaign script (`bench/v100/s111ncu2.sh`) documents this and names every profile
accordingly. (A no-match profile is also what let the app run past the `--kill` point into the
MTP decode deadlock, §11.)

Measured launches (4 K prefill; % of peak). 16 K nsys context in brackets (launches / total /
avg per launch):

| kernel (nsys class) | dur / launch | SM | mem | DRAM | L2 | verdict |
|---|---|---|---|---|---|---|
| `dequant_gu` (133 K / 6.5 s / 48 µs) | 56.9–58.7 µs | 9.3–9.5 % | 48–49 % | 16.8–19.4 % | 48–49 % | **latency-bound** — L2/DRAM feed dominates, SMs mostly idle |
| `dequant_flat` (150 K / 3.1 s / 21 µs) | 9.9–10.7 µs | 0.03 % | 0.57 % | 0.09 % | 0.57 % | **launch-overhead-bound** — ~150 K ~10 µs kernels; cost is scheduling, not work |
| `magma_sgemmEx` (3 456 / 4.4 s / 1.3 ms) | 1.9 / 482.1 µs | 11.4–11.8 % | 9.6 % | 2.5 % | 1.0–1.8 % | **latency/occupancy-bound** — even the big 482 µs GEMM pushes the SMs to ~11 % |
| `Kernel2` 128×128 cutlass (dense/PLE GEMM, 672 launches 16 K) | 1.35 / 837.7 µs | 76.1–79.4 % | 70.7–73.7 % | 20.0–20.6 % | 52.0–53.9 % | **efficient** — compute+memory balanced, near the roofline |
| `Kernel2` 64×64 cutlass (small GEMM, 40 K launches 16 K) | 25.0–28.7 µs | 15.6–17.9 % | 33.9–38.8 % | 31.4–33.6 % | 24.7–28.0 % | **memory-bound** |
| `volta_s884gemm` 256×128 (large GEMM) | 793.0 µs | 80.8 % | 64.0 % | 14.2 % | 41.3 % | **efficient** — compute-bound at ~81 % of SM peak |
| `volta_s884gemm` 64×64 (small GEMM) | 138.7 µs | 48.2 % | 87.0 % | 12.7 % | 61.3 % | **memory-bound** |
| `splitKreduce` (split-K tail) | 6.1–8.9 µs | 2.7–10.2 % | 6.1–41.5 % | 6.1–41.5 % | 3.1–17.4 % | small, memory-bound |
| `attn_chunk` (QSA; 6 437 / 2.7 s / 419 µs @16 K) | 39.7–59.5 µs (4 K) | 9.1–10.4 % | 14.9–20.8 % | 1.7–2.9 % | 2.5–2.7 % | **latency-bound** — per-launch time scales with context (419 µs @ 16 K) |
| `gdn_rec` (288 / 1.4 s / 4.8 ms @16 K) | 5.3 µs (4 K) | 22.9 % | 22.6 % | 4.5 % | 2.3 % | **latency-bound** |
| `bf16_f32_mmvf` (PLE; 41 K / 1.4 s / 34 µs) | 20.2–70.4 µs | 37.0–41.1 % | 72.6–85.3 % | 72.6–85.3 % | 29.8–32.3 % | **DRAM-bound** — the only kernel pushing ~85 % of DRAM peak |
| `swiglu_il` (133 K / 0.8 s / 6 µs) | 3.5–3.7 µs | 1.4–5.8 % | 2.6–10.5 % | 2.6–10.5 % | 1.4–5.2 % | tiny, latency-bound |

(Raw `.ncu-rep` in `Logs/gpu/s111ncu-*` — gitignored, regenerable via
`ncu --import` — with the full per-launch pivot appended to
`Logs/gpu/s111ncu3-campaign.log`.)

**The roofline picture** — three distinct populations:

1. **Efficient large dense GEMMs** (cutlass 128×128, volta 256×128: SM 76–81 %, mem 64–74 %) —
   the PLE/dense projections on big matrices already run near the SM+DRAM roofline. Little headroom
   here except the DRAM term.
2. **DRAM-bound medium kernels** (PLE `bf16_f32_mmvf` at 85 % DRAM, volta/cutlass 64×64 small
   GEMMs, split-K tails) — bandwidth-limited; the PLE projection is the single hardest
   DRAM consumer in the prefill.
3. **Latency-bound small/irregular kernels** (the whole MoE + QSA + GDN path: SM 9–23 % on
   dequant, expert GEMM, attention and recurrence, plus ~150 K ~10 µs `dequant_flat` / 133 K
   ~4 µs `swiglu` elementwise launches) — the SMs sit mostly idle while these stream.

**Corrects the earlier assumption**: the dequant and expert-GEMM kernels are *not*
compute-bound — SM throughput is ~9–11 % of peak on every MoE kernel measured, with L2/memory
activity at ~49 % (dequant) and ~10 % (GEMM). The "89 % GPU busy" from nsys is kernel *count*
(many small latency-bound launches plus the efficient dense GEMMs), not uniform efficiency. So
fusing dequant+GEMM (the deferred E5) attacks both the SM waste and the extra memory round-trip,
and cutting launch count (batching / megakernel) attacks the dominant scheduling cost; the dense
GEMM side is already near its roofline.

## 3. The Stage 1.10 baseline this stage beat against

Same-machine, same-build fresh runs (serve path, `s110.sh ctx`): 4 K r1/r2 = 13.99/13.22 s;
16 K = 40.40/39.74 s; 32 K = 76.74/76.50 s; decode 48.8–51.9 tok/s. (Stage 1.10 numbers:
13.2/39.6/73.7 s phase-A, 40.7/40.5/77.5 ctx — consistent within run-to-run noise ~±0.5 s.)

## 4. E4 — async routing-ids host round-trip (`STRATA_MOE_ASYNC_IDS=1`)

**Hypothesis (from the breakdown)**: the per-layer 80 KB routing-ids D2H goes to *pageable*
memory and is followed by a **full `cudaStreamSynchronize(m.cs)`** — the main thread stalls on the
*entire* m.cs tail (GDN/attention/GEMMs queued ahead of the D2H), ~2.6 ms per layer × 288 layers at
16 K ≈ 1.0 s of host blocking (the 1,008 ms breakdown line).

**Change (opt-in, `src/prefill/prefill.cpp`)**: pin `ids_host/slot_host/src_host`
(`cudaHostRegister` at init); issue the ids D2H **right after `route()`** (before the
shared-expert GEMMs, which don't need the ids) and wait for **only that copy via an event**
(`cudaEventSynchronize(ids_ev)`) instead of the whole-stream sync. Default OFF = legacy stream
order bit-for-bit.

**Result (16 K A/B)**: OFF 40.40/39.74 s → E4 40.48/40.23 s — **neutral (+0.08/+0.49 s, within
noise)**. Correctness: 32/32 golden, 256/256 byte-identical (int8 ref md5 `cdb7f7…`), decode
48.97 tok/s, cold r1 output bit-identical to OFF (md5 `6041c5…` both). **Why neutral**: the nsys
trace shows the GPU is *busy* during most of the 1,008 ms of host blocking (it has queued work
ahead of the D2H); the pure GPU-idle fraction was ~136 ms of D2H gaps — the ceiling E4 could have
recovered. Removing the host stall re-orders GPU work but doesn't remove GPU work.

## 5. E3 — MTP draft / prefill overlap (`STRATA_MTP_OVERLAP=1`)

**Hypothesis**: the 8 × ~171 ms serialized, host-synced MTP K/V builds at chunk boundaries (3.7 %
of wall) can overlap the next chunk's main-model prefill (the CE is idle in those windows anyway).

**Design (opt-in, `src/core/mtp.cpp` + `include/strata/core/mtp.hpp` + `src/program/generate.cpp`)**:
instead of the legacy per-sub-chunk loop (512 × 4-token sub-chunks, `cudaStreamSynchronize` after
each), capture **one CUDA graph for the whole 2048-token chunk** (32 sub-chunk records: mapped
host→device token/step/pos copies + `record_forward` K/V-only). Capture runs **synchronously at
`bind()`** (quiescent point: pool workers parked, no other thread launching; ~290–355 ms one-time).
Partial-boundary sizes (e.g. the final 2047) are lazy-captured on the main thread (~15 ms, one-off).
At each boundary the host fills the deep mapped buffers (same per-token values as the legacy
loop), does one D2D of the chunk's R rows (~84 MB) and **launches the graph without syncing** — the
~170 ms of drafter compute then overlaps the next chunk's prefill. Legacy path untouched when off.

**Two bugs found and fixed during bring-up** (both via gdb backtraces of repro runs):

1. **Capture-thread race (SIGABRT)**: the original design captured on a background thread. While
   any stream capture is active, a kernel launch on *another* stream (here the session's compute
   stream `m.cs`, from the main thread's `embed_row` → `iq_dequant_f32`) fails with
   `cudaErrorStreamCaptureUnsupported` (900); `iq_kernels.cu check()` did `std::exit(1)` on the
   main thread while the capture thread sat mid-`cudaLaunchKernel` → glibc `__owner == 0` mutex
   assert → SIGABRT. **Fix**: capture synchronously on the main thread at `bind()` (no other thread
   launches during it); lazy exact-n captures also run on the main thread. Empirically validated:
   main-thread capture during prefill (the lazy path) is safe.
2. **R-rows D2D race**: the unsynced overlap issues the R-rows D2D on the drafter stream while the
   next chunk's `gr_broadcast` rewrites the same `m.R` rows on the main stream (the legacy path was
   masked by its per-sub-chunk sync). **Fix**: the D2D runs on the **main-model stream** (the one
   that owns/rewrites `m.R` — in-stream order it finishes before the rewrite) and the chunk graph
   on the drafter stream waits for the copy via an event (`ov_copy_ev_`). Only the ~50–100 µs copy
   stays on the critical path; the graph's ~170 ms still overlaps.

**Result (A/B matrix, r1/r2)**:

| ctx | OFF | E3 ON | Δ r2 |
|---|---|---|---|
| 4 K | 13.99 / 13.22 s | 14.16 / 13.25 s | +0.03 s |
| 16 K | 40.40 / 39.74 s | 40.73 / 39.98 s | +0.24 s |
| 32 K | 76.74 / 76.50 s | 76.74 / 76.27 s | −0.23 s |

**Net-neutral** (r1 includes the one-time ~0.3 s capture). Correctness: 32/32 golden, 256/256
byte-identical, decode 50.2–51.9 tok/s, cold r1 bit-identical to OFF (16 K `6041c5…`, 32 K
`1fbe57…`). **Why neutral**: the §2.2 trace shows the SMs are busy with 18,540 kernels (249 ms)
during each 270 ms boundary gap — the draft work was already being done in that window; moving it
under the next chunk's prefill just re-assigns SM work on an ~89 %-utilized pipeline, so the
overlap saves the serialization but not the SM-time. (At 32 K the small negative is within noise.)

## 6. Why the candidates were neutral (the stage question, answered)

The 16 K prefill wall (36.6 s) is a **saturated fused pipeline**: ~89 % GPU-busy, with the MoE
sections pacing at the H2D feed rate (10.57 GB/s active CE) and the dequant+GEMM SM work filling
the rest. Three consequences, each confirmed by measurement:

1. **Overlap adds no SM capacity** (E3): the MTP draft's ~1.37 s of SM work is real work;
   overlapping it with the main prefill doesn't shorten a wall where the SMs are already ~89 %
   utilized — total SM work is unchanged, only its timing shifts. The 8 × 270 ms CE gaps that
   looked like "free overlap room" are actually SM-busy windows (draft kernels).
2. **The host stalls are mostly overlapped already** (E4): 1,008 ms of host-side sync, but the
   GPU is busy with queued work during most of it (only ~136 ms of pure D2H-idle gap) — the
   recoverable ceiling was ~0.15 s, below the ~±0.5 s measurement noise.
3. **The single CE is already well-fed** (E7): 8-slot ring + 7-deep lookahead → zero in-section
   DMA gaps; the 10.57 GB/s active rate is the per-CE TLP efficiency at ~2 MB transfers (a second
   CE measured the same rate in the same trace), and a second stream (`STRATA_TWO_STREAM_DMA`,
   parity-split ring slots) measured neutral (40.60/39.97 s vs 40.40/39.74 s).

The remaining real levers (ranked by the §2.3 roofline): the latency-bound MoE + QSA + GDN
launch storm (dequant+GEMM fusion, batching / megakernel, byte reduction via residency or
finer-quantized streaming blobs), and the DRAM-bound PLE projection — both bigger projects than
this stage's "one isolated change" scope. The efficient large dense GEMMs are already near the
roofline, so they are not the target.

## 7. Experiments not pursued (measured rationale)

- **E6 (QSA `attn_batch` 32→256)**: `attn_chunk` is the largest individual kernel class
  (419 µs avg per launch at 16 K) but the SOL profile (§2.3) shows it **latency-bound**
  (SM ~10 %, not compute-bound) — batching to 256 cuts per-token launch overhead, yet the SMs
  are already ~89 % utilized overall, so total SM work is unchanged, only its timing shifts —
  expected ≤ 0.2–0.5 s at 16 K, below the demonstrated noise floor; would need its own gate
  cycle. Deferred.
- **E2 (dequant on a side stream)**: dequant (9.4 s) overlaps with expert GEMM (4.7 s) only
  within the per-expert stage ring; same SM-contention argument as E3 → expected neutral, skipped
  after E3/E4/E7 all measured neutral (the stage stop rule).
- **Expert-residency growth**: VRAM-bound (peak 22.6 GiB at 262 K / ~15 GiB at 32 K + 9 K-slot
  cache); not a scheduling change.

## 8. What changed in the tree

- `include/strata/core/mtp.hpp`, `src/core/mtp.cpp` — E3: `set_overlap(chunk)` (mapped deep
  token/step/pos buffers + `ov_rin_` + capture stream/event), `set_main_stream`, synchronous
  chunk-graph capture at `bind()`, lazy exact-n capture, overlap branch of `prefill()` with the
  event-ordered R-rows D2D.
- `src/prefill/prefill.cpp` — E4: pinned ids buffers + `ids_ev` + early D2H + event wait
  (`STRATA_MOE_ASYNC_IDS`); E7: `copy2` stream + parity-split staging (`STRATA_TWO_STREAM_DMA`).
- `src/program/generate.cpp` — wires `mtp.set_overlap(o.prefill_chunk)` +
  `mtp.set_main_stream(main_stream)` (no-op unless the env flags are set).
- `bench/v100/` — `s111e3.sh`, `s111e4.sh`, `s111e7.sh` (gate + ctx A/B per experiment),
  `s111_breakdown.py` (10-category nsys parser), `s111ncu2.sh` (fixed SOL campaign:
  `--expert-cache 9000` + `wait_idle()` so application-replay passes see identical launch counts),
  `s110.sh` gained `EXTRA_ENV` passthrough for opt-in flags.
- All opt-in, default OFF; default decode/prefill behavior unchanged (proven by the cold-r1
  bit-identical outputs above and the fresh-engine gates).

## 9. Correctness / determinism summary

- Fresh-engine gates (E3, E4, E7, all-on; 16 K int8, production flags): 32/32 golden-prefix
  MATCH; 256/256 byte-identical determinism (md5 `cdb7f7d056f339ba704d3bb9620a1dec` — the Stage
  1.8/1.10 int8 reference) in every arm; decode 48.97–51.9 tok/s (target 49–50, within gate
  noise); all runs rc=0, no CUDA errors, no hangs.
- Serve path: cold r1 output **bit-identical to the OFF baseline** in every arm (16 K
  `6041c5…`, 32 K `1fbe57…`); warm r2 reproduces the OFF r2 output exactly for E7 and the
  combined arm (`a49423…`), and matches the pre-existing warm-residency caveat for the others.
- INT8 KV / QSA attention mechanism / production context size / Stage 1.9 sync: untouched
  (E4/E7 reorder *enqueues* and stream assignment only, opt-in; E3 touches only the MTP drafter).

## 10. Production state after the stage

- 32K production config restored (`strata-swift-iq3_xxs.json`, `--max-context 32768`, int8 KV,
  all opt-in flags OFF — defaults), `strata.service` restarted, port 8180 verified with a live
  request.
- Data/artifacts: `Logs/gpu/s111nsys-*.sqlite`, `Logs/gpu/s111ncu-*.ncu-rep` (both gitignored,
  regenerable),
  `Logs/benchmarks/s111e{3,4,7}-*.json` (gates), `Logs/benchmarks/s110-client.jsonl` (A/B runs),
  `Logs/gpu/s111e3-repro*.out` (E3 bring-up repros), `Logs/gpu/s111ncu2-campaign.log` +
  `s111ncu3-campaign.log` (SOL campaign runs, incl. the per-launch pivot).

## 11. Known quirks observed (not fixed here)

- Shared-engine warm r1/r2 fill runs diverge in output (expert-cache residency → GPU-hit-path
  numerics) — pre-existing since Stage 1.10; use cold r1 for bit-identical comparisons.
- ncu application-replay is fragile: expert-cache *auto* sizing drifts between replay passes when
  VRAM occupancy differs (fixed by a fixed slot count + idle-GPU precondition in `s111ncu2.sh`).
- Under ncu the MTP spec-decode phase can deadlock: a `wait_flag_ge` spin kernel holds the SMs at
  100 % (0 % memory) while the main thread sits in a `sched_yield` CUDA-sync loop and all 24 pool
  workers are futex-parked (observed ~45 min on the first campaign run; I/O counters frozen).
  Non-ncu 256-token decodes never show it → a CUPTI-interaction artifact on the flag-publish
  chain, not an engine regression. `--kill 1` avoids it entirely (kills the app once the profiled
  launches are done) — but only when the kernel filter actually matches: with a no-match filter
  the app runs to completion and can hit the deadlock.
- ncu's `-k` filter matches the *function* name base, not the nsys demangled name: the cutlass
  GEMMs are `Kernel2` (all tile variants), other cuBLAS GEMMs `volta_s884gemm_fp16_*`, split-K
  reduces `splitKreduce_kernel`. A no-match profile ends with ncu printing the full
  observed-kernel list in the run log — the definitive naming reference (§2.3).
- `iq_kernels.cu check()` still `std::exit(1)`s on a CUDA error (diagnostic, long-standing) — the
  E3 capture bug was found through it; a recoverable-error variant is a candidate for a later stage.
- The 10.57 GB/s active H2D rate (~80 % of PCIe3) at ~2 MB transfers is likely the per-CE TLP
  efficiency ceiling on this platform; larger contiguous staging (batched blobs) or a pinned
  host-side arena with `cudaMemcpyAsync` of bigger spans is the untested byte-side lever.
