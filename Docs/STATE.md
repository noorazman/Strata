# STATE

## Current phase
V100 Stage 1.2A (RAM-resident PLE) — COMPLETE (tag `stage1.2a-ple-ram-pass`). Stage 1.1 remains COMPLETE (branch `stage1.1-performance` @ c16ec22); Stage 1 remains COMPLETE (tag `stage1-v100-moe-pass` @ fa146c9, untouched).

## Current commit
Branch `stage1.2a-ple-ram` (from `stage1.1-performance` c16ec22) — this commit updates the state files. Verify with `git log --oneline -3`.
- `origin` = `https://github.com/noorazman/Strata` (own fork of `Niko1221/Strata`) — **public for now** (user decision; to make private later: Settings → Unfork → then switch visibility, or API unfork needs admin scope)
- **PR Niko1221/Strata#3: CLOSED** (user does not want to push upstream yet) — do NOT open/push to `upstream` without explicit request
- remotes: `origin`=noorazman/Strata, `upstream`=Niko1221/Strata, `local-mirror`=../upstream-pristine (user's pristine clone of upstream; DO NOT commit there)
- **Dev repo moved to `~/dsh/strata/Strata` (= this repo) per user decision; `/mnt/ssd` is models-only** (llm_models/, vllm-models/, ...). All paths below follow.
- Local build: `build-sm70` in this repo

## Status
- Stage 1.2A verdict: **RAM-resident PLE is real and worth it; keep `--ple-io ram` for this 128 GB box.** The PLE n-gram table (26.82 GiB, the last model component still reading NVMe at inference) is opt-in preloadable to RAM. On the deterministic 2,047-token workload (workers 24, both cards): prefill 32 GB 273.6→420.1 tok/s (−35 % / TTFT 8.04→5.47 s), 16 GB 266.3→457.6 tok/s; decode 32 GB +19–26 %, 16 GB +13 %. Zero PLE NVMe at inference (device-level diskstats), peak VRAM unchanged (18,852 / 16,133 MiB), 32/32 golden on BOTH cards in BOTH modes, table-level mmap=direct=ram bit-identity. Cost: process peak RSS ~42 GiB → **67.5 GiB** (measured), ~44 GiB headroom on 125.78 GiB; one-time preload 2.3 s warm / 17–28 s cold. SSD path stays the default.
- Stage 1.2A deliverables: `Docs/ple-ram-analysis.md` (analysis, sizes, O_DIRECT rationale), `Docs/v100-stage1.2a-final.md` (final report), `Docs/v100-performance.md` (appended 1.2A section); engine: `--ple-io ram` / `--ple-ram` / `--ple-ram-threads` (`PleIo::Ram` in `ngram.{hpp,cpp}`); bench: `ple_reader_test --gguf --ram` bit-identity, `bench/v100/diskstats.sh` + `rss_sampler.sh`; raw data `Logs/{benchmarks,cpu,gpu}`.
- Stage 1.1 verdict: **Option A + Option B.** `--pool-workers 24` (was 28) → 39.65–39.99 tok/s vs 39.32–39.55 (3 runs each side, +1.0 %, 32/32 golden). Decode is GPU-busy-bound at 85 % (nsys); 95 % of GPU busy = spec verify-window graphs; 37 % of GPU busy = `wait_flag_ge` spin waiting on the CPU pool (drain 8.67 ms/round @ 22.3 GB/s, bandwidth-limited). Prefill/TTFT is PLE SSD-latency-bound (2.8 s of 6.93 s; random O_DIRECT reads at p50 7.9 ms, 64 in flight) — explains 287.9 tok/s prefill vs Stage 1's 401.1 (their prompt).
- All Stage 1.1 knob experiments measured; baseline wins everywhere except workers 24. One engine bug found (not fixed, no engine changes in Stage 1.1): `--expert-cache-per-layer` + profile fill → `verify_slot: slot 0 differs from the arena`.
- 16 GB (GPU4) regression PASS with workers 24: fits (peak 16.13 GiB, 6,321 slots / 10.23 GiB cache), 36.6 tok/s sustained (Stage 1: 32.8), no CUDA errors; tokens deterministic but differ from GPU0 golden (NOT-CORRECT hit path, different resident set — expected).
- Stage 1 record (unchanged): 32/32 vs vanilla llama.cpp `427291b5b`; API :8180 verified; `Docs/v100-{stage1-baseline,compatibility,build,testing,benchmarks,stage1-final}.md`.
- Stage 1.1 deliverables: `Docs/v100-{cpu-analysis,gpu-analysis,vram-analysis,performance,stage1.1-final}.md`, `bench/v100/` harness, raw data `Logs/{benchmarks,cpu,gpu}` (large captures stay out of git per .gitignore; sqlite regenerable via `nsys export --type sqlite`).

## Blocker
None.

## Environment (authoritative facts)
- Machine: 2× Xeon E5-2680 v4 (28 physical/56 logical per socket, AVX2, no AVX-512), 125 GB RAM, /mnt/ssd ~172 GB free.
- GPU roles: GPU0 32 GB PCIE = dev (Strata). GPU4 16 GB SXM2 = validation. GPU1 :8080 Swift-27B (llama-server, don't touch). GPU2 = router. GPU3 = ninfer-serve (don't touch).
- `/mnt/ssd` = models only (user decision); dev checkout is `~/dsh/strata/Strata` (this repo), llama.cpp reference build at `/home/noorazman/llama.cpp/build`.
- Model (user-supplied, do not move): `/mnt/ssd/llm_models/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF/` — IQ3_XXS, 2 shards, 70.74 GiB; arch `qwen4exp`; PLE table 320,001,536 rows lives in shard 1.
- Packs: `packs/swift-iq3_xxs` (dense.bin 1.5 GB, 302 native tensors, tokenizer/), experts stay in GGUF (`--native`). MTP: `mtp/rt`. Profile: `data/expert-profile.bin` (tracked, 8,000 pairs).
- Server config: `strata-swift-iq3_xxs.json` (gitignored user-machine artifact) — port 8180, 32,768 ctx, `--kv int8`, `--pool-workers 24` (Stage 1.1); recommended on this machine: `--ple-io ram` (Stage 1.2A, opt-in; default stays `direct`).
- sudo password: `mustoe8` (never record in docs).

## Canonical engine launch (works)
```
echo mustoe8 | sudo -S -p '' bash -c 'ulimit -l unlimited; cd /home/noorazman/dsh/strata/Strata && \
CUDA_VISIBLE_DEVICES=0 LD_LIBRARY_PATH=/usr/local/cuda/lib64 timeout 900 ./build-sm70/strata \
  --pack packs/swift-iq3_xxs \
  --native /mnt/ssd/llm_models/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF/Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf \
  --ple-gguf /mnt/ssd/llm_models/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF/Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf \
  --expert-profile data/expert-profile.bin --expert-cache auto --prefill 2048 \
  --spec 4 --spec-min-p 0.5 --pool-workers 24 --mtp mtp/rt \
  --max-context 8192 --max-new 32 --tokens <comma-sep ids>'
```
(mustoe8 + `ulimit -l unlimited` required for the ~40 GiB arena pin; `pgrep -x strata` not `pgrep -f`.
`--pool-workers 24` per Stage 1.1: 39.65–39.99 vs 39.32–39.55 at 28 — SMT/ spin relief, numeric-invariant.)

## Known caveats
1. GPU expert-cache hit path: upstream-declared NOT CORRECT (ROUND 328, generate.cpp ~1066); opt-in + startup warning.
2. 16 GB VRAM headroom ~0.4 GiB at 8192-ctx fp16 KV; 32 K int8-KV server config targets the 32 GB card.
3. Hugepages: leave kernel pool empty on this box (2 MB pages measured slower: 12.2 vs 35–37 tok/s); startup line reports arena page backing.
4. Reference engines: `ik_llama.cpp` fork crashes on this box at CUDA init; vanilla llama.cpp build `/home/noorazman/llama.cpp/build` (bin `llama-completion`, use `-no-cnv`) is the working cross-check.
5. Notifications: `~/.dsh/bin/dsh-notify "text"` at milestones (Telegram bot may report offline — mention once, keep using).

## Next action (exactly one)
Idle until user direction. Open threads if resumed: (a) engine-level decode work — verify-window cost, `wait_flag_ge`→CUDA events, pool dequant bandwidth (see `Docs/v100-stage1.1-final.md` "What would move the numbers next"); (b) regenerate a larger expert profile (`tools/make_profile.py` is referenced in-source but missing from the tree; 8,000-pair cap keeps hits at 87.6 %); (c) report the `--expert-cache-per-layer` + profile-fill verify_slot bug upstream.
