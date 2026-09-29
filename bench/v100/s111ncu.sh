#!/bin/bash
# Stage 1.11 — ncu SpeedOfLight profiles of the top prefill kernels (4K prefill, GPU0, service stopped).
# One ncu run per kernel family (this ncu build takes a single -k).  ~2 min each.
set -u
cd /home/noorazman/dsh/strata/Strata
SHARD=/mnt/ssd/llm_models/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF/Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf
PACK="--pack packs/swift-iq3_xxs --native $SHARD --ple-gguf $SHARD --expert-profile data/expert-profile.bin"
TOK=$(cat Logs/gpu/s111ncu4k-tokens.txt)

profile() { # label regex [skip] [count]
  local label="$1" k="$2" skip="${3:-0}" count="${4:-3}"
  echo "[$(date +%H:%M:%S)] ncu $label"
  echo mustoe8 | sudo -S -p '' bash -c "
    ulimit -l unlimited
    cd /home/noorazman/dsh/strata/Strata
    export CUDA_VISIBLE_DEVICES=0 LD_LIBRARY_PATH=/usr/local/cuda/lib64
    /usr/local/cuda/bin/ncu -f -o Logs/gpu/s111ncu-$label \
      --section SpeedOfLight \
      -k 'regex:$k' --launch-skip $skip --launch-count $count \
      ./build-sm70/strata $PACK --kv int8 --max-context 32768 \
      --prefill 2048 --spec 4 --spec-min-p 0.5 --pool-workers 24 --mtp mtp/rt --expert-cache auto \
      --max-new 16 --tokens '$TOK' > Logs/gpu/s111ncu-$label.run.log 2>&1
    echo RC=\$?
  "
  echo "[$label] rc=$?"
}

profile dqgu    "dequant_gu_kernel"
profile dqflat  "dequant_flat_kernel"
profile magma   "magma_sgemmEx" 20 3
profile ck32    "cutlass_70_wmma_tensorop_s161616gemm_f16_32x32"
profile ck16    "cutlass_70_wmma_tensorop_s161616gemm_f16_16x16"
profile ck64    "cutlass_70_tensorop_s884gemm_f16_64x64"
profile attn    "attn_chunk_kernel" 0 2
profile gdn     "gdn_rec_kernel" 0 1
profile ple     "bf16_f32_mmvf"
profile swiglu  "swiglu_il"
profile splitk  "splitKreduce|volta_s884gemm_fp16_64x128" 20 6
echo "S111 NCU DONE $(date +%H:%M:%S)"
