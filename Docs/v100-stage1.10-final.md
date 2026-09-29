# V100 Stage 1.10 — Prefill / TTFT / Long-Context Investigation (measurement stage)

Production target: **ONLY GPU0, Tesla V100 32 GB** (SM70), `strata.service` on port 8180,
Swift-1.5-Qwen3.8-Flash-Next GSQ/RCO IQ3_XXS. Stage 1.9 state was the baseline; this stage
added opt-in instrumentation (`STRATA_TTFT`, `STRATA_WAIT_FENCE`), measured TTFT end-to-end,
localized the Stage 1.9 flag-A visibility stall by phase, and tested context up to 262 K.
**No default-path engine changes** — the binary's decode/prefill behavior is unchanged
(gates prove it, §7). Two env-gated features were added, both OFF by default.

Stage goal (verbatim): *Investigate why Open WebUI appears to wait before receiving the
first thinking token, and determine whether Strata can realistically support up to 262K context.*

---

## 1. Method & instrumentation

### 1.1 New opt-in instrumentation (committed this stage)

| knob | where | what it emits | default |
|---|---|---|---|
| `STRATA_TTFT=1` | `src/program/generate.cpp` | startup `strata ttft config: {json}` (max_context, kv, layer split, kv bytes/token, kv capacity, session bytes, QSA state bytes) + one `strata serve ttft: {json}` line **per request** to stderr: `t_parse_us`, `t_prefill_start_us` (session-zero + sync + slot borrow), `t_prefill_end_us`, `t_ready_us` (lent-slot refill), `t_first_window_us`, `t_first_token_us`, `t_first_token_out_us` (pipe flush lag), `prefill_chunks`, `prefill_ms`, `prefill_ple_ms`, `experts_streamed/experts_dma/experts_resident`, `experts_host_ms` | OFF |
| `STRATA_TTFT=1` | `serve/server.py` | `strata serve ttft: {json}` to stderr per HTTP request: `t_json_ms`, `t_prepare_ms` (tokenize), `t_gen_write_ms`, `t_first_engine_tok_ms`, `t_first_sse_ms`, `t_first_reasoning_ms`, `t_first_content_ms`, `t_done_ms` (all relative to request start) | OFF |
| `STRATA_WAIT_FENCE=1` | `src/kernels/cuda/verify_kernels.cu` | `__threadfence_system()` every 1024 polls inside the `wait_flag_ge` spin loop (explicit period also accepted via `STRATA_WAIT_FENCE=<n>`) | OFF |
| `STRATA_WAIT_ITERS=1` | `src/core/verify.cpp` | wait-iteration max per (l,grp,slot) at Verifier destruction (Stage 1.9-style diagnostics) | OFF |

The serve path was verified to use **no `wait_flag_ge` on the prefill path** (code inspection
of `sp.run` + trace confirmation, §4: 0 waits before the first doorbell).

### 1.2 Bench tooling (new files, `bench/v100/`)

- `s110_client.py` — OpenAI-compatible client with per-SSE-line timestamps
  (t_send / t_first_byte / t_first_data / t_first_reasoning / t_first_content / t_done,
  per-token arrivals → steady decode over the last 70 %, text md5, template overhead from
  usage). Appends to `Logs/benchmarks/s110-client.jsonl`.
- `s110.sh` — `ttft` (Phase A matrix), `ctx <C>` (per-context long-lived server: overhead
  probe, 1K warmup, two fill-prompt runs with 256 max-new, 2 Hz VRAM CSV, meminfo, md5 check),
  `ctxgate <C>` (32-tok golden prefix + 256-tok ×2 byte-identical on a fresh engine,
  `--kv int8 --max-context C`), `fence` (Phase F A/B).
- `s110nsys.sh` — nsys serve-mode captures (16K int8/32768, fence off/on; 256-tok fp16/8192
  fence on) with `--cuda-event-trace=false` (NSYS 2025.1.3 device-side event-completion
  trace stalls on the prefill path's per-chunk `cudaEventRecord` — see §9).
- `s110_trace.py` — phase attribution: buckets every `wait_flag_ge` into **prefill /
  round-0 (first token) / decode** on the verify-window stream (identified by doorbell
  count), with A/B/C breakdown and per-round round-head-A distribution.
- `s110_summary.py` — per-context sweep table from client + engine ttft jsonl + VRAM CSVs.
- `s110_sweep.sh` — driver: `ctx` + `ctxgate` for C = 4096 … 262144.
- `corpus-265k.ids` — one 262,987-token BPE corpus (2016 paragraphs); any token cut
  round-trips exactly (0/200 random cuts drifted) — used to build exact-N prompts.

### 1.3 Run environment

strata.service stopped for all profiling; GPU0 only; `sudo ulimit -l unlimited` (40 GiB
pinned arena); PLE 26.82 GiB RAM-resident; other services (ninfer:8001, llama-server:8080)
untouched. The box **rebooted at 01:47 mid-sweep** (strata.service auto-started on boot,
claiming the GPU; stopped again). The sweep was resumed from 8192, so all sweep data below
is post-reboot, single consistent session.

---

## 2. TTFT measurement (Task 1) — "where does the wait before the first thinking token come from?"

### 2.1 Client-side TTFT matrix (production 32K config, WARM server, `s110.sh ttft`)

| prompt | tokens | TTFT thinking-ON | TTFT thinking-OFF | steady decode |
|---|---|---|---|---|
| 4K | 4,148 / 4,108 | **13.15 s** | **11.93 s** | 55.2 / 52.9 tok/s |
| 8K | 8,244 / 8,204 | **21.80 s** | **20.49 s** | 53.1 / 54.6 tok/s |
| 16K | 16,436 / 16,396 | **39.55 s** | **38.39 s** | 52.1 / 53.5 tok/s |
| 32K | 32,632 / 32,620 | **73.66 s** | **73.17 s** | 61.7 / 57.2 tok/s |

Thinking on/off changes the template overhead by 52−12 = 40 tokens of prefill only —
**no measurable TTFT difference** (≈ 90 ms at 460 tok/s).

### 2.2 Engine-side stage decomposition (same runs, `strata serve ttft` lines)

| prompt | parse | ingest (session-zero+sync+borrow) | **GPU prefill** | prefill tok/s | refill (lent slots) | first window | T-line flush lag |
|---|---|---|---|---|---|---|---|
| 152 (1st request) | <0.1 ms | 1.1 ms | 2,232 ms | 68 (cold cache) | 163 ms | 133 ms (cold) | 12 µs |
| 4,148 | <0.1 ms | 1.2 ms | 10,413 ms | 398 | 162 ms | 25.0 ms | 13 µs |
| 8,244 | <0.1 ms | 1.4 ms | 19,102 ms | 432 | 162 ms | 25.0 ms | 12 µs |
| 16,436 | <0.1 ms | 2.6 ms | 36,829 ms | 446 | 162 ms | 25.5 ms | 14 µs |
| 32,632 | <0.1 ms | 4.1 ms | 71,096 ms | 459 | 162 ms | 25.4 ms | 15 µs |

Client-side overhead (request start → engine `t_arr`): JSON build + Python tokenize +
GEN-line write ≈ **2.3 s at 32K** (scales with prompt text size, not tokens).

**Answer to Task 1: the first-token delay is ~98 % GPU prefill.** At 32K:
71.1 s of 73.2–73.7 s. Everything else is small: HTTP+tokenize ≈ 2.3 s (client-visible,
server-side Python BPE), session reset ≤ 4 ms, expert-slot refill 162 ms (constant —
fixed lent-slot set, not prompt-dependent), first verify window 25 ms, engine→pipe
flush 15 µs. **The delay is NOT "API overhead" or "ingestion"** — it is the batched
MoE+attention prefill itself at a steady ~460 tok/s. The cold first request after server
start is a separate effect: prefill drops to ~36 tok/s while the expert cache fills
(69.7 % GPU hits), i.e. ~1.7 s for 62 tokens instead of ~0.14 s warm.

### 2.3 TTFT vs prompt length (linear regime)

Client TTFT is linear in prompt tokens with slope ≈ 2.17 ms/token (≈ 460 tok/s) from 4K to
32K; prefill rate saturates at ~450–460 tok/s by 8K (expert streaming becomes the rate
determinant, §3). Extrapolating the 32K slope: 64K ≈ 150 s, 128K ≈ 305 s, 262K ≈ 660 s —
confirmed by the sweep (§5): measured 151.7 / 312.3 / 675.8 s.

---

## 3. Prefill bottleneck (Task 1 / question 4)

nsys 16K serve capture (int8/32768, 16,384-token prompt, fence off): prefill phase =
1,195,701 kernels on the prefill stream, **32.76 s GPU-busy in a 36.6 s span (89 % busy —
the prefill is compute+streaming-bound, not launch-gap-bound)**. Top contributors:

| kernel | calls | GPU time | share | what it is |
|---|---|---|---|---|
| `dequant_gsu_kernel` | 133,176 | 6,452 ms | **19.7 %** | expert weight dequant (gate/up) |
| `magma_sgemmEx_kernel` | 3,456 | 4,439 ms | **13.5 %** | dense/MLP GEMM (MAGMA) |
| `Kernel2` (fused MoE) | 157,139 | 3,492 ms | **10.7 %** | per-token expert compute |
| `dequant_flat_kernel` | 149,999 | 3,126 ms | **9.5 %** | expert weight dequant (down) |
| `attn_chunk_kernel` | 6,144 | 2,674 ms | **8.2 %** | QSA prefill attention chunks |
| `gdn_rec_kernel` | 288 | 1,387 ms | 4.2 % | GDN linear-attention recurrence |
| `bf16_f32_mmvf_kernel` | 32,767 | 1,377 ms | 4.2 % | PLE native post-ops (per token) |
| `volta_s884gemm_fp16_*` | 103,694 | 2,731 ms | 8.3 % | fp16 GEMMs (projections) |
| `splitKreduce_kernel` | 140,299 | 967 ms | 3.0 % | GEMM split-K reduce |
| expert H2D (memcpy) | 107,479 ops | **190.11 GB** | — | expert weight streaming (overlapped) |

**The prefill long pole is MoE expert weight streaming + dequant.** 190 GB of expert
weights cross PCIe during a 16K prefill (≈ 18 s of DMA at ~10.5 GB/s, overlapped under
32.8 s of GPU work); dequant is 29 % of kernel time. The QSA attention chunk path is a
real but secondary 8.2 % (and it is the part that grows at long context, §5). Experts
streamed per request grows 44,409 (4K) → 1,581,576 (262K); the expert cache's resident
hit share *shrinks* 45 % (4K) → 39 % (262K, more unique experts at 8,000 slots). PLE native
post-ops: 5–8 % of prefill time (0.66 s @4K → 33.0 s @262K, RAM-resident at 1.3–1.9 GB/s).

Prefill rate vs context (sweep, §5): ~450 tok/s flat 4K→64K, 427 @128K, 392 @262K — the
long-context slowdown is the QSA chunk-attention + indexer work (selection width is
`min(n_kv, 2051)` capped, but the chunked QSA kernel still processes full-context chunks).

---

## 4. Flag-A phase attribution (Task 2) — "is the Stage 1.9 flag-A stall in prefill, first-token, or decode?"

`s110_trace.py` on the nsys serve captures (16K int8/32768, 16,384-token prompt + 64 decode
tokens; the verify-window stream is identified by doorbell count, 960 doorbells = 48
dispatches × 20 rounds):

| phase | waits | A / B / C total | round-head A |
|---|---|---|---|
| **prefill** (before first doorbell) | **0** | 0 / 0 / 0 | — |

(Phase A confirms the same at 32K: the prefill path contains no `wait_flag_ge` kernel —
verified in the 16K nsys capture and in code: `sp.run` uses only event syncs.)
| **round 0 (first token)** | 144 | A 6.0 ms (max 4.95) / B 10.9 / C 8.5 — total 25.3 ms in a ~25–30 ms first window | **4.95 ms** |
| **decode (rounds 1–19)** | 2,736 | A 206.7 / B 573.9 / C 112.3 ms | **20/20 rounds > 1 ms, mean 9.37 ms, max 11.9 ms** |

**Answer: the flag-A lag is a verify-window (decode-phase) phenomenon — 0 prefill waits.**
At the first token it is ~5 ms of a ~25 ms first window (the round-0 boundary visibility
lag, same mechanism as Stage 1.9's round-head A). In steady decode it is the round-head
A: ~9.4 ms every round (~10 % of the ~95 ms round). Note for this config (int8 KV +
expert-cache auto via the serve loop) flag **B** (CPU row staging) is the largest single
wait class — the CPU pool rows are on the critical path when the GPU has plenty of
resident experts; in the fp16/8192 generate workload of Stage 1.9 flag A dominated.

### 4.1 `STRATA_WAIT_FENCE=1` (the Stage 1.9 recommended experiment)

e2e A/B (`s110.sh fence`, 256-tok decode, golden + determinism gates both arms):

| config | OFF | ON | Δ e2e | output |
|---|---|---|---|---|
| fp16 / 8192 | 50.46 tok/s | 50.06 tok/s | **−0.8 %** | bit-identical, md5 `c1517d…` = GOLDEN |
| int8 / 32768 | 49.07 tok/s | 49.26 tok/s | **+0.4 %** | bit-identical, md5 `cdb7f7…` (int8 ref) |

Kernel-level (nsys, 256-tok fp16/8192, s19 workload, fence OFF = `s19-base.sqlite`):

| metric | OFF | ON (fence=1024) | Δ |
|---|---|---|---|
| round-head A mean | 9.53 ms/round | 8.54 ms/round | **−10 %** |
| round-head A max | 14.63 ms | 10.45 ms | **−28 %** |
| rounds > 1 ms | 111/111 | 111/111 | — |
| decode A total | 1,098.4 ms | 993.3 ms | −9.6 % |
| total wait | 1,672.3 ms | 1,602.0 ms | **−4.2 %** |

16K serve captures (int8/32768): round-0 A 6.0 → 5.5 ms (head 4.95 → 4.13 ms); decode
round-head A mean 9.37 → 8.97 ms; prefill still 0 waits both arms.

**Verdict: the periodic fence removes the *tail* of the round-head A visibility lag
(−10 % mean, −29 % worst) but not its mean body — the lag is a window-boundary effect:
the poll starts before the flag is published and the first reads of the new window still
hit the stale line. Net e2e ≈ 0 (−0.8 % / +0.4 %, within noise; the fence's own cost
offsets the visibility gain at 1024-period granularity).** Keep it opt-in as a diagnostic
and tail-latency knob (`STRATA_WAIT_FENCE=1`); do **not** make it the default yet. The
structural fix (publish the round-boundary flag earlier / pre-warm the line at window
head) remains the Stage 1.9 recommendation and is ~2,398 µs/tok of headroom if realized.

**Does flag-A affect TTFT? (question 5): only the round-0 share — ~4–5 ms of a 73 s
32K TTFT (0.006 %). It is a decode-round-latency problem, not a TTFT problem.**

---

## 5. Long-context sweep (Task 3) — 4K → 262K, int8 KV, production flags

`s110_sweep.sh`: per context, a fresh engine at `--max-context C --kv int8` serves
overhead probe → 1K warmup → fill-prompt run r1 (C−52−256−8 tokens) → r2 (warm); then
`ctxgate C` = fresh-engine 32-tok golden prefix + 256-tok ×2 byte-identical pair.
Fill prompt = exact-N cut of the 262,987-token corpus (round-trip verified).

| max-ctx | fill prompt | prefill (wall) | prefill tok/s | ingest | refill | first win | client TTFT r1 | decode r1 / r2 | **peak VRAM** |
|---|---|---|---|---|---|---|---|---|---|
| 4,096 | 3,780 | 8.63 s | 438 | 0.92 ms | 173 ms | 25.3 ms | 14.08 s | 50.1 / 57.1 tok/s | **18,736 MiB** |
| 8,192 | 7,876 | 17.33 s | 454 | 1.4 ms | 173 ms | 25.5 ms | 22.80 s | 50.4 / 57.1 tok/s | **18,800 MiB** |
| 16,384 | 16,068 | 35.34 s | 455 | 2.3 ms | 173 ms | 25.3 ms | 40.72 s | 51.8 / 48.9 tok/s | **18,926 MiB** |
| 32,768 | 32,452 | 71.88 s | 451 | 4.4 ms | 174 ms | 25.8 ms | 77.46 s | 50.2 / 51.3 tok/s | **19,178 MiB** |
| 65,536 | 65,220 | 146.0 s | 447 | 8.3 ms | 172 ms | 32.7 ms | 151.7 s | 51.9 / 53.9 tok/s | **19,668 MiB** |
| 131,072 | 130,756 | 306.4 s | 427 | 16.5 ms | 175 ms | 28.6 ms | 312.3 s | 56.1 / 66.6 tok/s | **20,646 MiB** |
| **262,144** | **261,828** | **668.7 s** | **392** | **32.2 ms** | **179 ms** | **30.1 ms** | **675.8 s** | **54.8 / 46.8 tok/s** | **22,604 MiB** |

All numbers warm-server r1 except where noted; r2 = warm repeat. No CUDA errors, no OOM,
no hangs at any context; engine logs clean.

### 5.1 Memory at 262K (the question that matters)

- int8 KV code: **1056 B/token/layer × 12 QSA layers = 12,672 B/token** → 3.09 GiB at
  262,144 (measured `kv_cache_capacity_bytes` = 3,321,888,768).
- Measured peak VRAM 262K: **22,604 MiB** (r1 = r2; bench.py repeat run: 22,582 MiB)
  against 32,768 MiB → **10.1 GiB headroom (31 % of the card)**.
- 32K → 262K VRAM growth = 22,604 − 19,178 = 3,426 MiB for 229,376 added tokens =
  15.4 KiB/token — matches KV (12.4 KiB/token) + RoPE table (64 MiB at 262K, first QSA
  layer only) + idx_pooled growth. The estimate from the Stage 1.10 planning notes
  (22.3–22.5 GiB peak) was confirmed.
- **The memory component that bounds context is the QSA int8 KV cache
  (12,672 B/token).** GDN state is context-independent (fixed recurrent buffers);
  expert cache (8,000 slots / 12.93 GiB) and PLE (RAM) are context-independent.
- Arithmetic extrapolation (not tested, from the measured 32K→262K growth): 524K ≈
  26.2 GiB (~6.4 GiB headroom) — feasible; 1M ≈ 33.5 GiB — just over the card →
  **262K is the demonstrated practical max on a 32 GB V100, and it fits comfortably**.

### 5.2 Decode vs context

No systematic decode penalty: 48.9–57.1 tok/s at 4K→64K, 56.1/66.6 at 128K,
54.8/46.8 at 262K. The r2 dip at 262K (46.8 vs 54.8 r1) tracks expert-cache eviction
(1.58M expert activations vs 612K resident at 262K — the 8,000-slot cache thrills
harder as the working set grows; the engine's pre-existing "GPU hit path is
approximate" warning applies). GDN layers are context-independent by construction; the
QSA layers see the full context but are sparse (selection capped at 32,768 cells =
131,072 tokens at idx_block 4), so decode cost per token grows only mildly with context.

### 5.3 262K is not destabilizing

262K ran two fill-prompt requests (r1/r2) plus a fresh-engine bench.py pair back-to-back
with stable VRAM (22,604 / 22,582 MiB), no CUDA errors, clean engine shutdown on stdin
EOF. 128K is the "comfort" point (20,646 MiB, 12.1 GiB headroom) if headroom matters for
future feature growth; 262K is the demonstrated max.

---

## 6. Correctness / determinism (Task 4)

Per completed config (all 7 sweep contexts + Phase F + Phase A):

- **32/32 golden**: fresh-engine 32-tok run at `--kv int8 --max-context C` on the
  canonical 29-token prompt — first-32 prefix MATCH of the int8 golden reference
  `GOLDEN_32` at **every context 4096…262144** (`s110.sh ctxgate`).
- **256/256 deterministic**: fresh-engine 256-tok ×2, byte-identical,
  md5 `cdb7f7d056f339ba704d3bb9620a1dec` (the int8/32768 Phase F reference) at **every
  context**.
- **Fill-prompt determinism** (256-tok generation from the 32,452- and 261,828-token
  fill prompts, fresh engine per run, `--tokens-file`): **IDENTICAL at 32K and at 262K**.
- **Shared-engine fill-prompt r1 vs r2 diverged at every context** — expected: the
  expert cache's residency differs between the two runs of one long-lived engine
  (the pre-existing "GPU hit path is NOT CORRECT" caveat: timing is real, output is
  approximate when the cache is involved). Fresh-engine pairs are the determinism
  contract; they pass everywhere.
- **Phase F**: both fence arms bit-identical to their respective references (fp16/8192
  = GOLDEN_MD5 `c1517d…`, int8/32768 = `cdb7f7…`), 32/32 + 256/256 green both arms.
- No CUDA errors, no hangs, no divergence in any gate run; all rc=0.

---

## 7. The ten numbered answers (stage goal)

1. **Current 32K TTFT**: **73.2–73.7 s** client-side (thinking off/on; 32,632/32,620
   prompt tokens; warm server). Engine: 71.1 s prefill + 0.16 s refill + 0.03 s first
   window + <0.005 s ingest; ~2.3 s HTTP/tokenize. Cold first request after server start:
   ~1.9 s for 62 tokens (36 tok/s cold-cache prefill).
2. **TTFT vs prompt length**: **linear**, slope ≈ 2.17 ms/token (≈ 460 tok/s prefill)
   from 4K (13.2 s) through 32K (73.7 s); 64K/128K/262K measured 151.7 / 312.3 / 675.8 s
   — the slope holds (392 tok/s at 262K, the long-context QSA-attention slowdown).
3. **Where the first-token delay occurs**: **~98 % is GPU prefill** (the batched
   MoE+attention prefill at ~460 tok/s). The rest: client HTTP+tokenize ~2.3 s (32K),
   session reset ≤ 4 ms, expert-slot refill 162 ms (constant), first verify window
   25 ms, pipe flush 15 µs. It is *not* API overhead or ingestion.
4. **Prefill bottleneck**: **MoE expert weight streaming + dequant** — 190 GB H2D
   during a 16K prefill (~18 s of DMA overlapped under 32.8 s of 89 %-busy GPU work);
   dequant 29 % of kernel time, GEMM 25 %, expert compute 11 %, QSA chunk attention
   8.2 %, GDN recurrence 4.2 %. The ~450 tok/s ceiling is the streaming+dequant rate;
   the 262K dip to 392 tok/s is QSA long-context chunk attention.
5. **Does the flag-A lag affect TTFT?**: **No, not meaningfully** — 0 prefill waits;
   ~4–5 ms of the ~25 ms first-token window (round-0 boundary); 0.006 % of 32K TTFT.
   It is a decode-round phenomenon (~9.4 ms/round ≈ 10 % of round time).
6. **262K memory feasibility**: **Yes** — measured peak **22,604 MiB < 32,768 MiB
   (10.1 GiB headroom)**; int8 KV 3.09 GiB at 262K (12,672 B/token); no OOM/CUDA errors;
   stable decode.
7. **Max practical context**: **262,144** (demonstrated, 31 % VRAM headroom). The
   bounding component is the QSA int8 KV cache at 12,672 B/token; 524K would fit by
   arithmetic with ~2.5 GiB headroom (not tested); 1M does not. 128K (20.6 GiB) is the
   comfort point.
8. **Decode as context grows**: **stable ~49–57 tok/s** to 128K; 46.8–54.8 at 262K
   (r2 dip = expert-cache eviction at the 1.58M-activation working set). No systematic
   context penalty (GDN context-independent; QSA sparse, selection-capped).
9. **Recommended next optimization**: (a) **prefill expert streaming/dequant** — the
   TTFT long pole (29 % dequant + 190 GB H2D): overlap dequant with DMA, or persist
   dequantized weights for hot experts (VRAM-perf trade); (b) **round-head flag-A**
   (decode −10 % rounds): publish the round-boundary flag earlier / pre-warm the flag
   line at window head (~2,400 µs/tok theoretical, Stage 1.9); `STRATA_WAIT_FENCE=1`
   is a partial (tail-only) mitigation, e2e-neutral — keep opt-in; (c) minor: move
   server-side BPE off the request path or stream-parse it (~2.3 s at 32K).
10. **What NOT to optimize**: the QSA attention kernel (8.2 % of prefill, 0.8 % of a
    decode token — Stage 1.8), GDN recurrence (4.2 %, context-independent), the 162 ms
    refill (fixed slot set), the 25 ms first verify window, the int8 KV code itself
    (1,056 B/token/layer is near-minimal for 64-group scales), PLE (RAM-resident, 5–8 %
    of prefill), expert-cache slot count (8,000/12.93 GiB is balanced), and the
    MTP-draft path. FA (FlashAttention) on this V100 is explicitly not a TTFT or
    long-context lever at this model's sparse attention.

---

## 8. What changed in the tree

Committed this stage (all opt-in or test-only; default behavior unchanged):

- `src/program/generate.cpp` — `STRATA_TTFT` startup config line + per-request serve
  ttft line (with `PrefillStats`: chunks, ms, PLE ms, experts streamed/dma/resident,
  host ms).
- `src/kernels/cuda/verify_kernels.cu` — `wait_flag_ge_kernel` takes a fence period;
  `__threadfence_system()` every N polls; `wait_fence_period()` reads `STRATA_WAIT_FENCE`
  (unset/0 → off; 1 → 1024; else explicit).
- `src/core/verify.cpp` — announces the fence state at capture start; `STRATA_WAIT_ITERS`
  wait-iteration slot report.
- `serve/server.py` — `STRATA_TTFT` HTTP-side stage line (tokenize/GEN-write/first-token/
  first-SSE/reasoning/content/done).
- `bench/v100/` — `s110_client.py`, `s110.sh`, `s110nsys.sh`, `s110_trace.py`,
  `s110_summary.py`, `s110_sweep.sh`, `corpus-265k.ids`, `fill-32452.tok`,
  `fill-261828.tok`; `bench.py` now passes `--tokens-file` through natively (argv
  strings are capped at 128 KB — a 262K-token prompt is 1.86 MB).
- `Docs/v100-stage1.10-final.md`, `Docs/STATE.md`, `Docs/CHANGELOG.md`.

Production 32K config (`strata-swift-iq3_xxs.json`, 32,768 / int8) unchanged as the
control.

## 9. Known quirks observed (not fixed here)

- NSYS 2025.1.3 device-side CUDA-event-completion trace stalls on the prefill path's
  per-chunk `cudaEventRecord` (engine blocked in libcupti `cudaEventRecord` for minutes).
  Workaround: `--cuda-event-trace=false` on s110 nsys captures (analysis uses kernel +
  memcpy rows only).
- First SIGINT to the Python serve process can leave the main thread parked in
  `threading.Event.wait()` (signal delivered to a worker thread); a second SIGINT/SIGTERM
  recovers; `s110.sh stop_server` already falls back to KILL; the engine exits on stdin EOF.
- `PrefillStats.ms_total` accumulates across requests of one server process (probe +
  warmup + r1 + r2) — the per-request prefill number in this doc is the wall span
  `t_prefill_end_us − t_prefill_start_us`; PLE is quoted as the cumulative percentage.
- The box rebooted at 01:47 mid-sweep; sweep data is post-reboot (resumed at 8192).

## 10. Data & artifacts

- `Logs/benchmarks/s110-client.jsonl` — all client TTFT runs (Phase A + sweep).
- `Logs/benchmarks/s110-A-engine-ttft.jsonl` — Phase A engine stage lines.
- `Logs/benchmarks/s110-ctx-<C>-engine-ttft.jsonl` + `s110-ctx-<C>-engine.log` +
  `s110-ctx-<C>-vram{1,2}.csv` — per-context sweep engine/VRAM data (C = 4096…262144).
- `Logs/benchmarks/s110-ctx<C>-{g32,det1,det2}.json` — per-context gates.
- `Logs/gpu/s110-f-{fp16-8192,int8-32768}-{off,on}.{json,out}` — Phase F A/B.
- `Logs/gpu/s110nsys-16k-{off,on}.sqlite` (291/296 MB), `s110nsys-256-fence.sqlite`
  (124 MB) + `*.run.log` — nsys captures; reference: `s19-base.sqlite` (fence OFF, 256 tok).
- `bench/v100/s110_trace.py` output (this doc §4) and `s110_summary.py` output (§5).
- `Logs/gpu/s110-sweep*.log`, `s110-fill-det.log`, `s110-fill{32768,262144}-det{1,2}.out`
  — sweep + fill-determinism logs.

**Next (Stage 1.11, NOT started)**: prefill expert-streaming/dequant overlap (answer 9a)
and/or the round-head flag-A structural fix (answer 9b). Stop after this stage.
