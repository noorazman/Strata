# V100 Release 1 — Production Freeze & Validation

Qwen3.8 Flash Next MoE (Swift IQ3_XXS, GSQ-RCO) on 1× Tesla V100 32 GB PCIe (GPU0, SM70).
Release 1 is a **productionization and validation milestone**, not an optimization stage:
it freezes the latest validated Stage 1.x baseline and proves it reproducible from a clean start.
Remaining MoE GEMM/DMA optimizations (Stage 1.17 follow-ups: deeper DMA staging, smem-staged G2b,
grouped D GEMM) are deferred until **after Release 4** per the mission stop rule.

## Verdict

**RELEASE 1 READY — FREEZE THIS BASELINE.** All 14 release gates PASS.

- Release commit: the commit tagged **`v100-release1`** on branch `stage1.3-expert-pool-sync` (this
  doc, STATE/CHANGELOG, unit update, gate harness, raw gate data); the engine baseline it builds from
  is `a374342` (Stage 1.17 HEAD) with **no engine code change** in the release commit.

## 1. Exact Release 1 configuration

**Engine** (production JSON `strata-swift-iq3_xxs.json`, served by `strata.service`, port 8180):

```
./build-sm70/strata --pack packs/swift-iq3_xxs
  --native /mnt/ssd/llm_models/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF/Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf
  --ple-gguf <same shard>
  --expert-profile data/expert-profile.bin      # 8,000 pairs (E8 14K profile NOT used)
  --expert-cache auto                            # -> 8,000 sized slots
  --prefill 2048 --spec 4 --spec-min-p 0.5
  --pool-workers 24 --mtp mtp/rt
  --max-context 32768 --kv int8
```

- `--ple-io ram` is the program default since Stage 1.2B (no flag).
- `--pcie-frac 0.2` is the native-pack default since Stage 1.3.
- **Environment (systemd unit `strata.service`, the only Release-1 change to production):**
  `CUDA_VISIBLE_DEVICES=0`, `LD_LIBRARY_PATH=/usr/local/cuda/lib64`,
  **`STRATA_MOE_DQ_WIDE=1`** — the one validated Stage 1.x improvement enabled (Stage 1.14:
  bit-identical output, prefill-only, dequant kernel time −49.6 % @16K / −49.7 % @32K).

**Build:** clean `build-sm70` (wiped + reconfigured + rebuilt for Release 1): CMake 4.4.3 (repo
venv), Ninja, gcc 11.4, CUDA 12.9.86, `-DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_CUDA=ON
-DSTRATA_NATIVE_EXPERTS=ON -DSTRATA_BUILD_TESTS=OFF -DCMAKE_CUDA_ARCHITECTURES=70
-DFETCHCONTENT_BASE_DIR=/home/noorazman/dsh/strata/fetch`. Fatbin verified **sm_70 only**
(39 ELF sections, no other archs). `ctest -R "parity|selftest|hostref"`: 19/20 (the single failure
is the pre-existing environmental `ple_parity` — missing Q2_0 fixture, red on every stage since 1.4).

## 2. Proven optimizations (enabled)

All of Stages 1.1–1.7 landed as **program defaults** (measured wins, frozen by later stages);
they are on with no flags. The only non-default production switch is the 1.14 env var:

| stage | optimization | state in Release 1 |
|---|---|---|
| 1.1 | `--pool-workers 24` (was 28) | default flag in JSON |
| 1.2A/B | PLE n-gram table preloaded to RAM (`--ple-io ram` default) | program default |
| 1.3 | `--pcie-frac 0.2` (decode +11.5–12.3 %) | native-pack default |
| 1.4 | fused single-phase CPU expert pool (decode +4.5–5.1 %) | program default (pool mode 7) |
| 1.5 | ExpertPool hybrid spin-then-futex parking | program default |
| 1.6 | arena NUMA pin + `STRATA_POOL_ASYNC` (default ON), WC flags | program defaults |
| 1.7 | `STRATA_GR_NORM_SPLIT` (default ON) | program default |
| 1.14 | MoE prefill dequant wide memory ops | **`STRATA_MOE_DQ_WIDE=1` in the service env** |
| — | `--spec 4 --spec-min-p 0.5 --mtp mtp/rt` (speculative decode) | JSON flags (Stage 1 era) |

## 3. Experimental optimizations (verified OFF)

| flag | stage | verdict | Release 1 |
|---|---|---|---|
| `STRATA_MOE_DEQUANT_GEMM_FUSE` (E5) | 1.12 | 1.78–1.83× slower prefill | OFF (default) |
| E8 `--expert-cache 14000` + 14K profile | 1.13 | TTFT −4.3/−5.4 %, decode neutral, VRAM −9.2 GB margin | OFF (8,000-slot production cache) |
| `STRATA_MTP_OVERLAP` / `STRATA_MOE_ASYNC_IDS` / `STRATA_TWO_STREAM_DMA` (E3/E4/E7) | 1.11 | net-neutral | OFF (default) |
| `STRATA_WAIT_FENCE` | 1.10 | e2e-neutral | OFF (default) |
| `STRATA_GR_DOWN_SPLIT` / `STRATA_ROUTE_SORT` (E1/E3) | 1.7 | +133/+100 ms | OFF (default) |
| `STRATA_MOE_GEMM_LT` | 1.15 | WASH | OFF (default) |
| `STRATA_MOE_GEMM_TC` (+`_MAXNE`) | 1.16 | WASH | OFF (default) |
| `STRATA_MOE_GEMM_GROUPED` | 1.17 | WASH (negative, +8.8/8.7 %) | OFF (default) |
| `STRATA_IQAVX2`, `STRATA_ALTFLAGC`, `STRATA_POOL_UNFUSE` | 1.4/1.6 | neutral/restore-only | OFF (default) |

Verification method: the live engine process `/proc/<pid>/environ` contains exactly `CUDA_VISIBLE_DEVICES=0`,
`LD_LIBRARY_PATH`, and `STRATA_MOE_DQ_WIDE=1` — no other `STRATA_*` vars; the engine's startup log
reports the 8,000-slot expert cache and int8 KV (see gate log below). All OFF-path code is
byte-identical to the Stage 1.16/1.17 verified leg-A baseline (those stages proved leg-A re-runs
reproduce the production anchor md5s exactly when the flags are off).

## 4. Release gates (results)

| # | gate | result |
|---|---|---|
| 1 | clean production build for SM70 | PASS — wiped `build-sm70`, reconfigured + rebuilt (log `build-sm70-r1.log`), fatbin = 39 ELF sections **all sm_70**, `ctest -R "parity\|selftest\|hostref"` 19/20 (sole failure = pre-existing environmental `ple_parity`, missing Q2_0 fixture; red on every stage since 1.4) |
| 2 | 32K / int8 KV production config starts | PASS — engine boots: PLE ram preload (320,001,536 rows / 26.82 GiB), `expert cache auto: 26.61 GiB free -> 8000 slots`, `expert cache 8000 slots, 12.93 GiB of VRAM` pre-filled + slot-0 verified, int8 KV, `/health` `max_context: 32768` |
| 3 | `strata.service` starts | PASS — `systemctl is-active strata.service` = active after the Release-1 unit install (boot ~95–105 s) |
| 4 | `/health` passes | PASS — `{"status": "ok", "max_context": 32768, "model": "swift-iq3_xxs", "images": false, "api_key": false}` |
| 5 | live API request | PASS — `POST /v1/chat/completions`: "count 1..5" → `1\n2\n3\n4\n5` (2.8 s); "27*43" → `1161` (3.2 s), valid reasoning + content, usage sane |
| 6 | 32-token golden prefix | PASS @16K + @32K — first-32 == `GOLDEN_32` (first token 271), `s18check.py --prefix` MATCH |
| 7 | 256-token determinism, fresh engines | PASS @16K + @32K — **3/3 fresh-engine runs byte-identical** (det1==det2==det3), md5 `cdb7f7d056f339ba704d3bb9620a1dec` at both contexts = the established Stage 1.10–1.14 int8 reference |
| 8 | 16K production validation | PASS — §5.1; r1 fill output **byte-identical to the 1.14–1.17 anchor** (md5 `6041c5f3…`) |
| 9 | 32K production validation | PASS — §5.2; r1 fill output **byte-identical to the 1.14–1.17 anchor** (md5 `1fbe577e…`) |
| 10 | no CUDA errors / OOM / hangs / unexpected GPU usage | PASS — zero CUDA-error strings in any Release-1 log (`Logs/benchmarks/r1g-*.log`, `s110-ctx-{16384,32768}-engine.log`, serve logs); no OOM (VRAM peaks = the 8K-baseline values, §5); every run rc=0, every engine exited (no stragglers); only GPU0 touched (GPU1's 32 GB usage = pre-existing llama-server tenant) |
| 11 | experimental flags OFF | PASS — live engine `/proc/<pid>/environ` contains exactly `CUDA_VISIBLE_DEVICES=0` + `STRATA_MOE_DQ_WIDE=1` (checked on both service starts); startup log shows the 8,000-slot cache (not 14,000) and the standard prefill kernel set (no LT/TC/G2/E5 paths) |
| 12 | performance vs production baseline | PASS — §5.3: r1 TTFT inside the 1.14–1.17 anchor band at both contexts, and the r1 fill-prompt outputs are byte-identical to the anchors |
| 13 | decode ~55 tok/s class | PASS — serve-mode 51.09/54.27 (16K r1/r2), 49.51/52.0 (32K r1/r2) = the documented 48–58 tok/s production class; det-gate reference §5.4 |
| 14 | reproducible from clean start | PASS — the entire campaign ran from a wiped build → `daemon-reload` → fresh service → fresh engines, and reproduced the anchor r1 outputs byte-for-byte; exact clean-start procedure in §6 |

## 5. Measured results (Release 1 build, GPU0, 2026-10-02)

### 5.1 Engine-level gates (fresh engines, `bench.py` canonical flags, `STRATA_MOE_DQ_WIDE=1`)

| context | gate | result |
|---|---|---|
| 16,384 | 32-tok golden prefix | MATCH (first token 271, first-32 == GOLDEN_32) |
| 16,384 | 256-tok det ×3 (fresh engines) | byte-identical 3/3, md5 `cdb7f7d056f339ba704d3bb9620a1dec` (= the 1.10–1.14 int8 reference) |
| 16,384 | decode (det gate) | 48.36 / 48.96 / 48.50 tok/s (3 runs) |
| 32,768 | 32-tok golden prefix | MATCH (first token 271, first-32 == GOLDEN_32) |
| 32,768 | 256-tok det ×3 (fresh engines) | byte-identical 3/3, md5 `cdb7f7d056f339ba704d3bb9620a1dec` |
| 32,768 | decode (det gate) | 49.44 / 46.34 / 47.44 tok/s (3 runs) |

All six runs rc=0, zero CUDA errors. Harness: `bench/v100/r1gates.sh`; raw data
`Logs/benchmarks/r1g-*.json|log`, `Logs/gpu/r1g-*.out|csv`, `Logs/gpu/r1gates-run.log`.

### 5.2 Serve-mode 16K / 32K production validation (`s110.sh ctx`, `EXTRA_ENV=STRATA_MOE_DQ_WIDE=1`)

| 16K fill (16,068 tok → 16,120 prompt) | r1 (cold) | r2 (warm) |
|---|---|---|
| client TTFT (s) | **37.98** | 37.08 |
| engine first-token (ms) | 32,667 | 32,358 |
| decode (tok/s) | **51.09** | 54.27 |
| fill r1 text md5 | `6041c5f35e174bf3d9a8a1e00c80e7bf` (= the 1.14–1.17 anchor, byte-identical) | — |
| r2 text md5 | — | `a4942388…` (documented pre-existing warm-r2 divergence, Stage 1.16; r2 md5 stable across days) |
| peak VRAM (MiB) | 18,926 (= the 1.13 8K baseline peak) | — |

| 32K fill (32,452 tok → 32,504 prompt) | r1 (cold) | r2 (warm) |
|---|---|---|
| client TTFT (s) | **71.50** | 70.72 |
| engine first-token (ms) | 66,044 (prefill 492.4 tok/s, 18 chunks) | 65,582 (495.8 tok/s, 34 chunks) |
| decode (tok/s) | **49.51** | 52.00 |
| fill r1 text md5 | `1fbe577e30c875c77dfadab6556b4476` (= the 1.14–1.17 anchor, byte-identical) | — |
| r2 text md5 | — | `2afb7cb3…` (matches the warm-r2 trajectory observed in the 1.17-era runs) |
| peak VRAM (MiB) | 19,178 (= the 1.13 32K baseline peak) | — |

The r2 engine `prefill_ms` field runs high at 32K (136.3 s vs 70.9 s) because r2's diverged
trajectory streamed ~2× more experts during the request (404,031 vs 214,978) — a session-state
effect of the warm-r2 divergence, consistent with Stage 1.13/1.16 findings; the anchor metric is
the r1 client TTFT.

### 5.3 Baseline comparison (Gate 12)

Established production anchors for the same wide-ON production config:

| | 1.14 (wide-ON measured) | 1.16 leg A (fresh production) | 1.17 leg A (fresh production) | **Release 1** |
|---|---|---|---|---|
| 16K r1 TTFT (s) | 37.72 | 38.71 (4-run avg) | 40.66 | **37.98** |
| 32K r1 TTFT (s) | 71.36 | 72.24 (3-run avg) | 76.75 | **71.50** |
| 16K r1 fill md5 | `6041c5f3…` | `6041c5f3…` | `6041c5f3…` | `6041c5f3…` ✓ byte-identical |
| 32K r1 fill md5 | `1fbe577e…` | `1fbe577e…` | `1fbe577e…` | `1fbe577e…` ✓ byte-identical |
| VRAM peak 16K/32K (MiB) | — | — | — | 18,926 / 19,178 (the 8K-baseline values) |

Both r1 TTFTs land inside the 1.14–1.17 anchor band (16K spread 37.7–40.7 s and 32K 71.4–76.8 s
across measurement days = the documented ~±1.5–2.7 % r1-TTFT noise band on this machine —
correctable-MCE stream + scheduler), and the Release 1 build reproduces the established r1
fill-prompt outputs **byte-for-byte at both contexts** — the strongest possible Gate-12/14 signal.

### 5.4 Decode class (Gate 13)

- det gate (fresh engine, the stable reference): **48.4–49.0 tok/s @16K, 46.3–49.4 tok/s @32K**
  — inside the 48.7–51.6 tok/s class measured by Stages 1.12–1.14 (32K runs sit at the low edge
  of the documented machine band; no run below the 46.3 floor seen in any stage).
- serve-mode (the production class): **16K r1/r2 = 51.09/54.27, 32K r1/r2 = 49.51/52.00 tok/s** —
  the same 48–58 tok/s class the pre-Release service showed (53.8/53.2/52.0/58.3/48.0 in its
  last requests) and the Stage 1.13/1.14 serve anchors (16K 52.21/54.47, 32K 49.62/53.55).

## 6. Reproducibility (Gate 14) — exact clean-start procedure

```bash
cd /home/noorazman/dsh/strata/Strata
git checkout v100-release1
# 1. clean build (Gate 1)
rm -rf build-sm70
.venv/bin/python3 .venv/bin/cmake -S . -B build-sm70 -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_CUDA=ON -DSTRATA_NATIVE_EXPERTS=ON \
  -DSTRATA_BUILD_TESTS=OFF -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
  -DCMAKE_CUDA_ARCHITECTURES=70 -DFETCHCONTENT_BASE_DIR=/home/noorazman/dsh/strata/fetch
.venv/bin/python3 .venv/bin/cmake --build build-sm70 -j 40
# 2. production service (unit carries STRATA_MOE_DQ_WIDE=1)
sudo cp Docs/strata.service /etc/systemd/system/strata.service && sudo systemctl daemon-reload
sudo systemctl restart strata && sleep 90 && curl -s http://127.0.0.1:8180/health
# 3. engine gates (service stopped; wide ON)
sudo systemctl stop strata
for C in 16384 32768; do
  python3 bench/v100/bench.py "r1g-g32-$C" --gpu 0 --workers 24 --max-new 32 --stats --kv int8 --max-context $C --env STRATA_MOE_DQ_WIDE=1
  python3 bench/v100/bench.py "r1g-det1-$C" --gpu 0 --workers 24 --max-new 256 --stats --kv int8 --max-context $C --env STRATA_MOE_DQ_WIDE=1
  python3 bench/v100/bench.py "r1g-det2-$C" --gpu 0 --workers 24 --max-new 256 --stats --kv int8 --max-context $C --env STRATA_MOE_DQ_WIDE=1
  python3 bench/v100/s18check.py --prefix "Logs/benchmarks/r1g-g32-$C.json" 32
  python3 bench/v100/s18check.py "Logs/benchmarks/r1g-det1-$C.json" "Logs/benchmarks/r1g-det2-$C.json"
done
# 4. serve-mode 16K/32K validation (service stopped)
EXTRA_ENV=STRATA_MOE_DQ_WIDE=1 bash bench/v100/s110.sh ctx 16384
EXTRA_ENV=STRATA_MOE_DQ_WIDE=1 bash bench/v100/s110.sh ctx 32768
sudo systemctl start strata
```

## 7. Known limitations

1. **Prefill/TTFT is MoE-weight-streaming bound** (~40 s cold @16K, ~75 s cold @32K): the 16K
   prefill streams ~170 GB of expert weights over PCIe (≈80 % of the copy-engine ceiling). The
   remaining levers (deeper DMA staging, grouped GEMM, smem-staged G2b) are the deferred
   post-Release-4 work from Stage 1.17's measurement.
2. **Expert profile is stale for general traffic**: the 8,000-slot production profile is only
   ~27 % execution-resident on the fill-prompt workload (Stage 1.13); E8 (14K rebuilt profile)
   cuts H2D −63.5 % for −4.3/−5.4 % TTFT but trades ~9.2 GB of VRAM margin — parked as the first
   post-Release candidate together with a production-traffic profile.
3. **Serve-mode r2 decode/session-state effects**: warm repeated-prompt runs decode faster
   (residency + MTP warm state); the fresh-engine det gate is the stable reference
   (Stage 1.16 also documented intermittent warm-r2 divergence as pre-existing).
4. **ROUND 328 hit-path numerics** (GPU expert-cache decode hit path) remain upstream-declared
   NOT CORRECT; the prefill hit path is numerics-invariant. Decode-tier residency at 8K is lower
   than at 14K (84.3 % hits measured at 14K in 1.13).
5. **Machine environment**: steady ~1/s correctable-MCE stream on MC_CHA bank 5 (pre-existing);
   the ~±1.5–2.7 % r1-TTFT run-to-run band is documented and accounted for in Gate 12.
6. 16 GB SXM2 cards (GPU4) are not part of Release 1 scope (32 GB GPU0 only, per the mission).

## 8. Artifacts

- Raw gate data: `Logs/benchmarks/r1g-*.json|log`, `Logs/gpu/r1g-*.out|csv`, `Logs/gpu/r1gates-run.log`
  (engine gates); `Logs/benchmarks/s110-ctx-{16384,32768}-engine.log`,
  `Logs/benchmarks/s110-ctx-{16384,32768}-engine-ttft.jsonl`,
  `Logs/benchmarks/s110-ctx-{16384,32768}-vram{1,2}.csv`,
  `Logs/gpu/s110-s110-ctx-{16384,32768}.serve.log`, `Logs/gpu/r1ctx-run.log`
  (serve-mode validation; today's client rows in `Logs/benchmarks/s110-client.jsonl`,
  labels `B-16384-*` / `B-32768-*`, ts 2026-10-02 10:47–10:56); `build-sm70-r1.log` (clean build).
- Gate harness: `bench/v100/r1gates.sh` (this release's engine-level gates).
- Service unit: `Docs/strata.service` (installed to `/etc/systemd/system/strata.service`,
  carries `Environment=STRATA_MOE_DQ_WIDE=1`).
