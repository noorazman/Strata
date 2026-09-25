# PLE / RAM Analysis — Stage 1.2A (Phases 1–5)

Date: 2026-09-26. Branch `stage1.2a-ple-ram` (from `stage1.1-performance` c16ec22;
Stage 1 baseline fa146c9 untouched). This document is the code-level analysis and
the RAM feasibility case. Results are in `Docs/v100-stage1.2a-final.md`.

## 1. What "PLE" is in this engine

`PLE` = the per-layer token embedding table, GGUF tensor
`per_layer_token_embd.weight` [160, 320001536], IQ4_NL, 90 bytes/row. It is the
**n-gram table**: for every token the engine hashes the (token, prev1, prev2)
context into 16 row indices (`ngram_rows`, `src/kernels/ngram.cpp`) and gathers
those 16 rows (head-slowest = 2560 floats = n_embd) at layer 1.

It is the ONLY tensor the engine reads from the GGUF at inference time;
everything else is served from the pack/arena/VRAM.

## 2. Data path (traced in code, not inferred)

```
shard1 GGUF (NVMe)
  └─ per_layer_token_embd.weight @ file offset 461,157,600, 28,800,138,240 B
       │
       │  GgufFile parse (header + tensor bounds check) — mapping released
       │
       ├─ PLE read (default: --ple-io direct)
       │    src/ngram/ple_reader.cpp  PleReader
       │      • 4 KiB O_DIRECT preads via platform::DirectFile
       │      • ONE Linux worker thread; pread completes inside submit
       │        (io_uring is future "phase L") — so "64 in flight" is a queue of
       │        up to 64 SERIALIZED preads, not 64 concurrent NVMe requests
       │      • 8-way row cache, default 1,048,576 rows (~95 MB)
       │    → 90 B rows into per-token / per-chunk host buffers
       │
       ├─ IQ4_NL dequantize on CPU (iq4nl_dequant_row)
       │
       ├─ cudaMemcpyAsync host → GPU (prefill: one 20.9 MB upload per 2046-token
       │    chunk; decode: one 10.2 KB upload per token)
       │
       └─ layer-1 PLE block on GPU (ple_block / ple postops)
```

Prefill (`src/prefill/prefill.cpp:279`): all T×16 rows of the chunk are computed
host-side, then `gather_batch` issues **one synchronous batched ticket** and
blocks until every row is read from SSD, then the whole embedding block is
uploaded. Decode (`src/core/layer.cpp:1021`): `issue` when the token id is known,
`collect` before layer 1 — 16 rows per token, overlapped with embedding + layer 0.
Verify windows (`src/core/verify.cpp:727`) also use `gather_batch`.

**The PLE rows are NOT cached between uses, are NOT mmap-resident (in direct
mode), are NOT discarded-and-reread: they are read from NVMe on demand, with a
95 MB in-process row cache that hits only ~0.8 % on real prompts** (rows of a
prompt are ~99.2 % unique — the n-gram hash of prose produces distinct rows).

## 3. Sizes (measured, this machine)

Component sizes as they exist in the current implementation:

| component | size | where at runtime |
|---|---:|---|
| PLE table (`per_layer_token_embd.weight`) | **28,800,138,240 B = 26.82 GiB** | NVMe (shard1, offset 461,157,600) — O_DIRECT, never in RAM in direct mode |
| PLE profile (`data/expert-profile.bin`, 8,000 ranked (layer, expert) pairs that pre-fill the VRAM cache) | 130,328 B | RAM (read once at startup) |
| expert metadata (pack index, native_experts.txt, mtp/rt draft layer) | ~790 MB (mtp) + small text files | RAM / VRAM (draft layer 808 MiB VRAM) |
| expert blobs (48 routed layers × 512 experts, native GGUF layout) | **39.97 GiB** (loaded at 8.15 GiB/s at startup, Stage 1.1 log) | **pinned RAM arena** (`cudaHostRegister` PORTABLE), loaded from the shards at startup via buffered reads |
| model weights: dense pack + native projections + token embedding | 1,466 MiB + 1,781 MiB + 260 MiB ≈ 3.5 GiB | mmap'd host memory (page-cache-backed) |
| other: KV cache, activations, MTP state | ~14 GiB VRAM (peak 18.9 GiB on GPU0) | VRAM |
| model total on disk | 70.74 GiB (shard1 39.79 GB + shard2 36.18 GB) | NVMe |

**PLE data ≠ model.** The PLE table is 26.82 GiB of the 70.74 GiB model; the
rest is experts (39.97 GiB, already RAM-resident), dense/projection weights
(~3.5 GiB, mmap) and MTP draft (786 MB).

Machine: 125.78 GiB total RAM (MemTotal 131,886,572 kB); at measurement
~14 GiB used by OS/other processes, ~72 GiB page cache, ~96 GiB available.
NVMe: `nvme0n1` = KINGSTON SNV2S1000G (931.5 GB), mounted at `/mnt/ssd` (the
model shards live there; `/` itself is a SATA disk, but nothing model-related
runs from it). Cold sequential O_DIRECT read measured: ~2.0 GB/s.

## 4. Why O_DIRECT (Phase 3)

Documented in-code, not a guess:

1. `include/strata/platform/direct_file.hpp`: *"The n-gram table (26.8 GiB)
   stays on the SSD and must never occupy RAM, including the OS file cache. A
   memory map cannot promise that; an unbuffered read can."*
2. `include/strata/kernels/ngram.hpp`: *"Direct is the default: unbuffered 4 KiB
   reads from the SSD, so the table never occupies RAM or the OS file cache."*
3. The original design targeted a **64 GB RAM** machine
   (`src/kernels/ngram.cpp`): *"the PLE shard is 26.8 GB against 63 GB of RAM
   that the 31.6 GB expert arena is also competing for, so they are not in the
   OS cache."* 26.8 GB table + ~32 GB arena left no headroom for the table to
   pollute the page cache with 4 KB random pages.
4. Consequence by design: the engine keeps its OWN bounded row cache
   (default 1,048,576 rows ≈ 95 MB, 8-way set-associative) and a bounded
   in-flight read queue, instead of relying on the OS cache. Windows uses
   `FILE_FLAG_NO_BUFFERING | FILE_FLAG_RANDOM_ACCESS | OVERLAPPED` for the same
   reasons (no cache, no read-ahead, async).

So the reason is **explicit memory-pressure policy + predictable,
engine-controlled caching**, NOT an I/O-scheduling quirk. It is the right
default on a 64 GB box; on this 128 GB box the constraint is much looser —
which is exactly what this stage tests. (Note the asymmetry: the 39.97 GiB
expert arena is deliberately loaded through the page cache at startup and
pinned — the "keep it out of RAM" policy was applied to the PLE table only,
because the table is 26.8 GiB of pure random 4 KB access with no reuse, while
the arena is 100 % reused.)

## 5. RAM feasibility (Phase 4)

Budget with the PLE table made RAM-resident (+26.82 GiB):

| item | GiB |
|---|---:|
| OS + other resident processes (llama servers etc.) | ~14 |
| Strata: expert arena (pinned) | 39.97 |
| Strata: PLE table (NEW, anonymous mmap) | 26.82 |
| Strata: mmap'd weights (pack + projections + embedding, page-cache-backed) | ~3.5 |
| Strata: KV host buffers, activations, pinned staging, engine overhead | ~0.7 (measured residual) |
| **Expected peak RSS of the process** | **~45–47** (estimate) → **67.5 GiB measured** (1 Hz VmRSS sampling of a RAM-mode run: arena 39.97 + PLE 26.82 + ~0.7) |
| Total RAM | 125.78 |
| **Headroom after peak** | **~44 GiB** (plus reclaimable page cache) |

The 26.82 GiB preload also warms the page cache with the table region; those
pages are reclaimable, so the worst-case footprint is still ~47 GiB process +
~14 GiB OS ≪ 125.78 GiB. **Full residency is feasible with a large margin** —
no swapping risk, no NUMA issue (single-socket machine), no hugepages needed
(the arena already runs on 4 KB pages: "MAP_HUGETLB unavailable").

Startup cost is the price: a one-time sequential 26.82 GiB read from NVMe
(≈ 2.0 GB/s cold O_DIRECT; buffered preload should be similar or better since
it can pipeline through the page cache) — measured in `v100-stage1.2a-final.md`.

## 6. Design options (Phase 5) and the chosen one

- **A. Full RAM preload** — at startup copy the whole table into an anonymous
  buffer; inference reads become memcpys. Simple, prompt-independent, removes
  ALL PLE I/O (prefill, decode, verify). Cost: +26.82 GiB RAM, one-time
  preload.
- **B. Hot PLE cache** — keep frequently accessed regions in RAM. The working
  set of one 2047-token prompt is ~33,000 rows = ~3 MB of data (20,376 unique
  4 KB pages ≈ 80 MB), but rows are ~99.2 % unique per prompt (0.8 % row-cache
  hits in the Stage 1.1 run), so a hot set does **not persist across prompts**:
  the first prompt of a session still pays the full I/O, and decode gains come
  only from repeated contexts. A page-granular ~100 MB cache would mostly help
  long single-process multi-prompt servers, not the prefill of a new prompt.
- **C. RAM-backed PLE mapping** — load into RAM and serve the existing
  access layer from it. This is what option A implements inside
  `PleTable` (the `PleIo::Ram` mode reuses `issue/collect/gather_batch`),
  i.e. A and C are the same thing at this layer.
- **D. Keep the SSD path** — status quo.

**Chosen for the experiment: A/C** — the n-gram working set is per-prompt and
~30 K unique rows, so only whole-table residency removes the prefill cost; and
26.82 GiB fits comfortably (Section 5). The experimental switch is
`--ple-io ram` (alias `--ple-ram`); the existing SSD path is untouched.

## 7. Expert-blob distinction (Phase 10, code-level)

The Stage 1.1 "11,439 expert-blob DMA" are **not** SSD reads. Expert blobs
live in the 39.97 GiB **pinned RAM arena** loaded at startup
(`src/core/expert_source.cpp`, `ArenaExpertSource`, `load_experts_gguf`).
At runtime the path is:

```
pinned RAM arena ──(cudaHostGetDevicePointer / async DMA)──▶ GPU
```

- cache hits: GPU computes from VRAM slots that were DMA'd from the arena;
- cache misses: the CPU pool reads the blob from the RAM arena, computes on CPU;
  the "expert blobs … read" counter counts arena→consumer blob hands.

So SSD → RAM → GPU exists only for the one-time 39.97 GiB arena load at
startup; the runtime expert path is RAM → GPU. PLE storage I/O (O_DIRECT at
inference time) and expert DMA (RAM→GPU at inference time) are separate
mechanisms and are measured separately below.
