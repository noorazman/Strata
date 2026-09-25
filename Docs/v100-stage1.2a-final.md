# V100 Stage 1.2A Final — RAM-Resident PLE

Branch `stage1.2a-ple-ram` (from `stage1.1-performance` c16ec22; Stage 1 baseline
fa146c9 untouched). Machine: 125.78 GiB RAM, 5× V100 (GPU0 = 32 GB PCIE dev,
GPU4 = 16 GB SXM2 validation), NVMe `nvme0n1` = KINGSTON SNV2S1000G at `/mnt/ssd`
(the model shards live there; `/` is a SATA disk with nothing model-related on it).
All runs via `bench/v100/bench.py` (workers 24, the Stage 1.1 production config),
NVMe counters captured device-level around every run (`Logs/cpu/*.diskstats-*`).
Code analysis, sizes and the feasibility case: `Docs/ple-ram-analysis.md`.

## What PLE is and how it moved (answer to the primary question)

PLE = `per_layer_token_embd.weight` [160, 320001536] IQ4_NL, **28,800,138,240 B
= 26.82 GiB**, inside shard 1 of the Swift-1.5 GGUF at file offset 461,157,600.
It is the n-gram table: 16 hashed row indices per token, gathered at layer 1.

Component sizes (measured, this machine):

| component | size | where at runtime (before this stage) |
|---|---:|---|
| PLE table | **26.82 GiB** | NVMe, O_DIRECT on demand, never in RAM (by design) |
| PLE profile (`data/expert-profile.bin`, 8,000 ranked pairs) | 130,328 B | RAM (read once at startup) |
| expert metadata (mtp/rt, pack index) | ~786 MB | RAM / 808 MiB VRAM |
| expert blobs (48 layers × 512 experts) | **39.97 GiB** | pinned RAM arena (loaded once at startup, buffered) |
| model weights (dense pack + native projections + embedding) | ~3.5 GiB | mmap'd host memory (page cache) |
| model total on disk | 70.74 GiB | NVMe (39.79 GB + 36.18 GB shards) |

**PLE data ≠ the model: the table is 26.82 GiB of the 70.74 GiB model.** The
experts (39.97 GiB) were ALREADY RAM-resident in a pinned arena — that path is
SSD → RAM at startup only, then RAM → GPU (Phase 10, below).

Current SSD path (traced in code, `Docs/ple-ram-analysis.md` §2):

```
shard1 (NVMe) → GgufFile parse (mapping released) → PleReader: 4 KiB O_DIRECT
preads on ONE worker thread (Linux submit() is a synchronous pread; "64 in
flight" is a queue of ≤64 SERIALIZED reads) → 90 B rows → IQ4_NL dequant →
H2D upload → GPU. Prefill: one synchronous batched ticket for the whole chunk.
Decode: issue/collect per token, 16 rows.
```

## Why O_DIRECT (Phase 3)

Documented in-code (`direct_file.hpp`, `ngram.hpp`): the table "must never
occupy RAM, including the OS file cache" — a 64 GB-RAM design policy where
26.8 GB table + ~32 GB expert arena left no headroom for the table to pollute
the page cache. The engine keeps its own bounded 95 MB row cache instead.
Predictable latency + explicit, engine-controlled caching — NOT an I/O
scheduling quirk. On this 128 GB box the constraint is looser; that is what
this stage tests.

## RAM feasibility (Phase 4, with measured peak)

| | GiB |
|---|---:|
| OS + other resident processes | ~14 |
| RAM-mode engine **peak RSS (measured, 1 Hz VmRSS sampling)** | **67.5** (arena 39.97 + PLE 26.82 + ~0.7 overhead) |
| Total | 125.78 |
| Headroom | **~44 GiB** + reclaimable page cache |

Comfortably feasible. SSD-mode process RSS is ≈42 GiB by the same accounting
(arena + non-PLE overhead; no PLE pages).

## Implementation (Phase 6)

`--ple-io ram` (alias `--ple-ram`), `--ple-ram-threads N` (default 16). Open
parses the GGUF exactly like Direct (validated bounds), preloads the table
region into an anonymous mmap buffer with buffered preads (multi-threaded,
timed, reported at startup and in the `--stats "ple ram:"` line), releases the
mapping like Direct. `issue/collect/gather_batch` serve rows by memcpy + the
same dequant and the same out-of-range zeroing. The SSD path is untouched;
default stays `direct`. Table-level test: `ple_reader_test --gguf SHARD --ram`
(bit-identity vs mmap and direct).

## Benchmark (Phases 7–9), GPU0 32 GB, workers 24, deterministic 2,047-token prompt

| metric | SSD (`direct`) | RAM (`--ple-io ram`) | Δ |
|---|---:|---:|---|
| time to first GPU work (cold) | 79.5 s | 113.4 s | +~20–24 s preload |
| **PLE preload** | — | 26.82 GiB: **2.33 s @ 11.49 GiB/s warm** (page cache); 16.9–28.5 s cold (1.0–1.6 GiB/s NVMe) | one-time |
| **prefill 2046 tokens** | 7,477.6 ms (273.6 tok/s) | **4,869.9 ms (420.1 tok/s)** | **−34.9 %** |
| prefill end-to-end (bench) | 267.4 tok/s | **406.6 tok/s** | **+52 %** |
| TTFT | 8,035.6 ms | **5,471.5 ms** | **−2,564 ms** |
| PLE host time | 2,818.7 ms (2,519.7 ms of it I/O blocked) | 299.5 ms (hash + copy + dequant + 20.9 MB upload — the new floor) | −89 % |
| **NVMe PLE reads during inference** | **20,376 reads / 85.2 MB** (p50 7.9 ms, p99 9.1 ms) | **0** | −100 % |
| decode, 256 tokens | 35.38 tok/s | **44.69 tok/s** | **+26.3 %** |
| peak VRAM | 18,852 MiB | 18,852 MiB | identical |
| GPU busy during prefill (10 Hz) | 53 % avg | 75–89 % avg | GPU no longer starved by PLE I/O |
| process peak RSS | ~42 GiB (accounted) | 67.5 GiB (measured) | +25.5 GiB |

In-RAM NVMe device-level check (Phase 9): the RAM run's total NVMe read delta
(70.9 GB) = cold arena load (40.0 GB) + preload (27.5 GB) + pack touches
(1.5 GB) — i.e. the PLE table touches the device ONLY during the startup
preload, and then never again. The SSD run's PLE bytes (85.2 MB prefill /
18.8 MB decode) disappear entirely. RAM mode is not "large RSS pretending";
the device counters prove the I/O moved.

Decode moved even though the expectation was "prefill only": in Direct mode
the host thread blocks on `collect` for the 16 serialized O_DIRECT reads per
token (1,271.7 ms blocked over the run, p99 read latency 64 ms), stalling the
token pipeline; RAM mode removes the block (pool time, spec stats and expert
cache stats are bit-identical between the arms). Bonus, not just prefill.

## 16 GB regression (Phase 14), GPU4, workers 24

| metric | SSD | RAM |
|---|---:|---:|
| prefill 2046 tokens | 7,683.9 ms (266.3 tok/s; bench 260.9 tok/s; TTFT 8.22 s; PLE 3,252.5 ms of which 2,974 ms I/O blocked, 20,376 reads / 85.2 MB) | **4,719.1 ms (433.6 tok/s; bench 418.4) cold / 4,470.9 ms (457.6 tok/s; bench 440.8) warm; TTFT 5.32 / 4.81 s** (+60–72 %) |
| decode, 256 tokens | 35.43 tok/s | **40.04 tok/s (+13 %)** |
| TTFT (29-token prompt) | 2,296 ms | 2,156 ms |
| peak VRAM | 16,133 MiB | 16,133 MiB (no VRAM cost) |
| expert cache | 6,321 slots / 84.6 % hits | identical (same resident set) |
| process peak RSS | ~42 GiB | 67.5 GiB (system RAM unchanged: 128 GB) |
| decode PLE NVMe | 4,744 reads / 19.9 MB | 0 |

The PLE RAM optimization costs nothing in VRAM; the 16 GB card's constraints
(6,321-slot cache, 16.13 GiB peak) are unchanged.

## Correctness (Phase 12)

- Table level: `ple_reader_test --gguf <shard1> --ram` — **bit-identical**
  across mmap, direct and ram for all 52,752 rows (20,000 random tokens + the
  full 2,047-token prompt), including the straddle/out-of-range cases the
  synthetic selftest covers.
- GPU0: SSD and RAM both reproduce the documented golden prefix **32/32**
  (the Stage 1 llama.cpp reference sequence), token-for-token identical
  between the two modes; the 2,047-token prompt's 16 generated tokens are
  also identical between modes.
- GPU4: SSD and RAM both match the Stage 1.1 GPU4-deterministic reference
  **32/32** (differs from GPU0 at token 6, expected: different resident set).
- No numerical differences of any kind: expert-cache hit/miss counts, spec
  acceptance, pool timing and peak VRAM are identical between the arms.

## Phase 10 — expert blobs, measured separately

The Stage 1.1 "11,439 expert-blob DMA" are **RAM → GPU**, not SSD reads. The
39.97 GiB expert arena is loaded ONCE at startup (buffered, page-cache-backed,
~5–30 s depending on cache state), pinned, and then every expert access is
arena → GPU (cache-hit DMA / CPU pool). Confirmed at the device level: during
the decode phase the SSD arm's NVMe delta is ~19 MB — the PLE rows only; the
experts never touch the NVMe at runtime. PLE storage I/O and expert DMA are
separate mechanisms and only the former is addressed here.

## Phase 11 — hot/cold (not needed)

A prompt's ~33,000 PLE rows are ~99.2 % unique (0.8 % row-cache hits), so a
hot set does not persist across prompts: first-touch I/O for a new prompt is
unchanged under a hot/cold design. Whole-table residency (26.82 GiB, fits with
~44 GiB headroom) is the simpler design that removes ALL PLE I/O, so the
hot/cold experiment is deferred — it would only matter for a long-running
multi-prompt server, where a ~100 MB page-granular cache could be added later
without touching this work.

## Recommendation

**Keep full RAM residency (`--ple-io ram`)** for this 128 GB machine:

- prefill +52 % (32 GB) / +53 % (16 GB), decode +26 % / +13 %, TTFT −2.5 s;
- zero PLE NVMe I/O at inference (device-level verified);
- bit-identical correctness on both cards, zero VRAM cost;
- cost: +25.5 GiB RAM (peak 67.5 GiB of 125.78 GiB) and a one-time preload
  (2.3 s warm / ≤28 s cold) — trivially worth it against ~2.5 s of prefill
  latency removed PER PROMPT plus the decode gain.

Keep the SSD path as the default (`direct`) so smaller-RAM machines keep the
Stage 1.1 behavior; use `--ple-io ram` in the run scripts for this machine.
Do NOT modify the decode engine (wait_flag_ge / CUDA events / verify window /
pool dequant / expert cache / dense support) — untouched in this branch.

## Next optimization (NOT started)

Prefill is now ~80 %+ GPU-busy: the remaining terms are the CPU expert pool
dispatch for the 2,046-token chunks (~300 ms PLE floor + pool drain) and the
launch/verify overhead. Candidate follow-ups, in order: (1) chunked prefill
PLE prefetch pipelined across chunk boundaries (the batched ticket already
supports it; the host could start the next chunk's rows while the GPU drains
the current chunk), (2) the already-measured but untouched pool/CPU side
(wait-flag overlap), (3) io_uring for the Direct path if it ever needs to stay
the default on a slow NVMe. Stop after this stage; no next step implemented.
