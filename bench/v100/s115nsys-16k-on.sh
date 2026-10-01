#!/bin/bash
# Stage 1.15 (LT-ON leg) — nsys capture of the E8 resident-growth prefill (16K, --expert-cache auto +
# the top-14,000 profile data/expert-profile-14k.bin).
#
# Same serve-mode single-request capture as Logs/gpu/s110nsys-16k-off.sqlite (the 8,000-slot
# production baseline, used for the H2D-byte comparison): engine --serve int8/32768,
# 16K-token corpus prompt + 64 max-new via stdin, STRATA_WAIT_ITERS=1 STRATA_TTFT=1.
# Used to verify the H2D byte reduction (stream 15 expert transfers) and kernel counts.
set -u
cd /home/noorazman/dsh/strata/Strata
NSYS=/usr/local/cuda/bin/nsys
SHARD=/mnt/ssd/llm_models/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF/Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf
PACK="--pack packs/swift-iq3_xxs --native $SHARD --ple-gguf $SHARD --expert-profile data/expert-profile.bin"
SERVE_FLAGS="--prefill 2048 --spec 4 --spec-min-p 0.5 --pool-workers 24 --mtp mtp/rt --expert-cache auto"

pgrep -x strata >/dev/null && { echo "FATAL: strata already running"; exit 1; }

python3 - <<'EOF'
ids = open("bench/v100/corpus-265k.ids").read().split()[:16384]
open("Logs/gpu/s115nsys-16k-gen.txt", "w").write(f"GEN 64 {','.join(ids)}\nQUIT\n")
print("wrote Logs/gpu/s115nsys-16k-gen.txt (16384 prompt tokens)")
EOF

echo "[$(date +%H:%M:%S)] CAP s115nsys-16k-on (baseline: auto + 8K profile)"
echo mustoe8 | sudo -S -p '' bash -c "
  ulimit -l unlimited
  cd /home/noorazman/dsh/strata/Strata
  export CUDA_VISIBLE_DEVICES=0 LD_LIBRARY_PATH=/usr/local/cuda/lib64
  export STRATA_WAIT_ITERS=1 STRATA_TTFT=1 STRATA_MOE_DQ_WIDE=1 STRATA_MOE_GEMM_LT=1
  $NSYS profile -o Logs/gpu/s115nsys-16k-on --force-overwrite true -t cuda \
    --cuda-event-trace=false --cuda-graph-trace=node \
    ./build-sm70/strata $PACK --kv int8 --max-context 32768 $SERVE_FLAGS --serve \
    < Logs/gpu/s115nsys-16k-gen.txt > Logs/gpu/s115nsys-16k-on.run.log 2>&1
"
echo "s115nsys-16k-on rc=$?"
$NSYS export --type sqlite --force-overwrite true -o Logs/gpu/s115nsys-16k-on.sqlite Logs/gpu/s115nsys-16k-on.nsys-rep 2>/dev/null
echo "export rc=$?"
