#!/bin/bash
# Stage 1.11 — ncu SpeedOfLight campaign, application-replay mode (kernel-replay mode fails on this app:
# "Failed to profile ... at 0%" on the first prefill kernel). One pass-set per kernel family.
set -u
cd /home/noorazman/dsh/strata/Strata
SHARD=/mnt/ssd/llm_models/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF/Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf
PACK="--pack packs/swift-iq3_xxs --native $SHARD --ple-gguf $SHARD --expert-profile data/expert-profile.bin"
TOK=$(cat Logs/gpu/s111ncu4k-tokens.txt)

# wait for any leftover ncu/strata from earlier test runs
while pgrep -f "nsight-compute" >/dev/null 2>&1 || pgrep -f "build-sm70/strata" >/dev/null 2>&1; do
  sleep 5
done
nvidia-smi -i 0 --query-gpu=memory.used --format=csv,noheader

# --expert-cache is a FIXED slot count, not auto: application replay re-runs the whole app per metric
# pass and requires an identical set of profiled launches in every pass.  auto-sizing from free VRAM
# drifts between passes when another engine shared the GPU (the first campaign run died on exactly
# that: "Unexpected number of profiled kernels").  9000 matches what auto selected for the clean
# 4K/32K-ctx runs ("resident 9063" in the 10:44 test run).
#
# --kill 1: terminate the app as soon as the requested --launch-count launches are profiled.  All 11
# profiles are PREFILL kernels, so the app is killed mid-prefill and never reaches the decode phase.
# Without it the app continues into the MTP spec decode and can DEADLOCK there under ncu: the
# wait_flag_ge spin kernel holds the SMs at 100 % while the main thread spins in a CUDA sync
# (sched_yield loop) and the pool workers sit futex-parked - observed 2026-09-29 on the first run
# (dequant_gu pass 1, ~45 min, GPU 100 %/0 % mem, rchar frozen).  Plain (non-ncu) 256-token decodes
# never show it, so it is an ncu-interaction artifact (CUPTI overhead on the flag-publish chain),
# not an engine regression.
wait_idle() {
  while :; do
    local mib
    mib=$(nvidia-smi -i 0 --query-gpu=memory.used --format=csv,noheader,nounits)
    pgrep -f "build-sm70/strata" >/dev/null && { echo "[$(date +%H:%M:%S)] waiting for strata to exit"; sleep 10; continue; }
    [ "$mib" -le 1536 ] && return 0
    echo "[$(date +%H:%M:%S)] GPU busy ($mib MiB)"; sleep 10
  done
}

profile() { # label regex [skip] [count]
  local label="$1" k="$2" skip="${3:-0}" count="${4:-2}"
  if [ -n "$SKIP_DONE" ] && [ -s "Logs/gpu/s111ncu-$label.ncu-rep" ]; then
    echo "[$(date +%H:%M:%S)] skip $label (report exists)"
    return
  fi
  wait_idle
  echo "[$(date +%H:%M:%S)] ncu $label"
  echo mustoe8 | sudo -S -p '' bash -c "
    ulimit -l unlimited
    cd /home/noorazman/dsh/strata/Strata
    export CUDA_VISIBLE_DEVICES=0 LD_LIBRARY_PATH=/usr/local/cuda/lib64
    /usr/local/cuda/bin/ncu -f -o Logs/gpu/s111ncu-$label \
      --section SpeedOfLight --replay-mode application --kill 1 \
      -k 'regex:$k' --launch-skip $skip --launch-count $count \
      ./build-sm70/strata $PACK --kv int8 --max-context 32768 \
      --prefill 2048 --spec 4 --spec-min-p 0.5 --pool-workers 24 --mtp mtp/rt --expert-cache 9000 \
      --max-new 16 --tokens '$TOK' > Logs/gpu/s111ncu-$label.run.log 2>&1
    echo RC=\$?
  "
}

# Kernel-name gotcha (found by the 2026-09-29 "no matching kernels" dead-ends): ncu's -k filter
# matches the "function" name base, NOT the nsys-style demangled name.  The cuBLAS/cutlass GEMM
# kernels that nsys shows as `void cutlass::Kernel2<cutlass_70_wmma_...>` are named **`Kernel2`**
# in ncu (every tile variant shares the name), the cuBLAS GEMMs as `volta_s884gemm_fp16_*`, and
# the split-K reduces as `splitKreduce_kernel` (no cublasLt:: prefix).  When a filter matches
# nothing, ncu prints the full observed-kernel list at the end of the run log - use it to pick
# the right name.  (The app then runs on and can deadlock in the MTP decode under ncu - another
# reason --kill 1 matters: it only fires when the count was actually profiled.)
profile dqgu    "dequant_gu_kernel"
profile dqflat  "dequant_flat_kernel"
profile magma   "magma_sgemmEx" 20 2
profile gemm1   "Kernel2" 0 2          # early cutlass GEMM launches (warmup class, e.g. 128x128 tile)
profile gemm2   "Kernel2" 500 2        # bulk cutlass GEMM (the 32x32/16x16-tile class)
profile volta   "volta_s884gemm" 0 2   # the second GEMM class (cuBLAS volta kernels)
profile splitk  "splitKreduce" 0 2     # split-K reduce tail (cuBLAS)
profile attn    "attn_chunk_kernel" 0 2   # QSA chunk attention
profile gdn     "gdn_rec_kernel" 0 1      # GDN recurrence
profile ple     "bf16_f32_mmvf" 0 2       # PLE projection
profile swiglu  "swiglu_il" 0 2           # swiglu

# The ncu CSV is long-form: one row per (kernel-id, section, metric), and kernel names contain
# commas, so pivot with python (quote-aware csv), one line per kernel launch: duration, SM %,
# memory %, DRAM %, L2 %.
cat > /tmp/s111ncu-pivot.py <<'PYEOF'
import csv, re, sys
data, order = {}, []
with open(sys.argv[1]) as f:
    for row in csv.DictReader(f):
        kid = row.get("ID", ""); kn = row.get("Kernel Name", "")
        mn = row.get("Metric Name", ""); v = row.get("Metric Value", "")
        if kid not in data:
            data[kid] = {"name": kn}; order.append(kid)
        d = data[kid]
        if mn == "Duration": d["dur_us"] = v
        elif mn == "Compute (SM) Throughput": d["sm%"] = v
        elif mn == "Memory Throughput": d["mem%"] = v
        elif mn == "DRAM Throughput": d["dram%"] = v
        elif mn == "L2 Cache Throughput": d["l2%"] = v
for kid in order:
    d = data[kid]
    short = re.sub(r"unnamed>::|\(.*", "", d["name"])
    print(f'{short} | dur_us={d.get("dur_us","?")} sm%={d.get("sm%","?")} mem%={d.get("mem%","?")} dram%={d.get("dram%","?")} l2%={d.get("l2%","?")}')
PYEOF
for f in Logs/gpu/s111ncu-*.ncu-rep; do
  [ -e "$f" ] || continue
  echo "=== $f"
  echo mustoe8 | sudo -S -p '' /usr/local/cuda/bin/ncu --import "$f" --csv 2>/dev/null > /tmp/s111ncu-import.csv
  python3 /tmp/s111ncu-pivot.py /tmp/s111ncu-import.csv
done
echo "S111 NCU2 DONE $(date +%H:%M:%S)"
