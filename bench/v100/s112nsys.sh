#!/bin/bash
# Stage 1.12 — nsys capture of the E5 fused dequant+GEMM prefill (16K, STRATA_MOE_DEQUANT_GEMM_FUSE=1).
#
# Same serve-mode single-request capture as Logs/gpu/s110nsys-16k-off.sqlite (1.10/1.11 baseline):
# engine --serve int8/32768, 16K-token corpus prompt + 64 max-new via stdin,
# STRATA_WAIT_ITERS=1 STRATA_TTFT=1.  Used to verify: kernel counts (dequant+GEMM launches replaced
# by the fused kernels), expert-GEMM time, and that H2D volume is unchanged (E5 touches no DMA).
set -u
cd /home/noorazman/dsh/strata/Strata
NSYS=/usr/local/cuda/bin/nsys
SHARD=/mnt/ssd/llm_models/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF/Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf
PACK="--pack packs/swift-iq3_xxs --native $SHARD --ple-gguf $SHARD --expert-profile data/expert-profile.bin"
SERVE_FLAGS="--prefill 2048 --spec 4 --spec-min-p 0.5 --pool-workers 24 --mtp mtp/rt --expert-cache auto"

python3 - <<'EOF'
ids = open("bench/v100/corpus-265k.ids").read().split()[:16384]
open("Logs/gpu/s110nsys-16k-gen.txt", "w").write(f"GEN 64 {','.join(ids)}\nQUIT\n")
print("wrote Logs/gpu/s110nsys-16k-gen.txt (16384 prompt tokens)")
EOF

echo "[$(date +%H:%M:%S)] CAP s112nsys-16k-e5 (STRATA_MOE_DEQUANT_GEMM_FUSE=1)"
echo mustoe8 | sudo -S -p '' bash -c "
  ulimit -l unlimited
  cd /home/noorazman/dsh/strata/Strata
  export CUDA_VISIBLE_DEVICES=0 LD_LIBRARY_PATH=/usr/local/cuda/lib64
  export STRATA_WAIT_ITERS=1 STRATA_TTFT=1 STRATA_MOE_DEQUANT_GEMM_FUSE=1
  $NSYS profile -o Logs/gpu/s112nsys-16k-e5 --force-overwrite true -t cuda \
    --cuda-event-trace=false --cuda-graph-trace=node \
    ./build-sm70/strata $PACK --kv int8 --max-context 32768 $SERVE_FLAGS --serve \
    < Logs/gpu/s110nsys-16k-gen.txt > Logs/gpu/s112nsys-16k-e5.run.log 2>&1
"
echo "s112nsys-16k-e5 rc=$?"
$NSYS export --type sqlite --force-overwrite true -o Logs/gpu/s112nsys-16k-e5.sqlite Logs/gpu/s112nsys-16k-e5.nsys-rep 2>/dev/null
echo "export rc=$?"
