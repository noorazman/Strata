#!/bin/bash
# Stage 1.7 E4 — measure QSA attention (NO Flash Attention implementation).
#  (1) --gpu-stages: per-layer-kind mixer (GDN vs QSA) + 5-stage split on the captured graph.
#  (2) --gpu-only-full: the true per-token GPU floor (pre+post+head, no pool).
#  (3) e2e A/B x2 rounds: default (fast attn + fast select) vs --no-fast-attn vs --no-fast-select,
#      256-token decode, GPU0, canonical tokens, drop_caches between arms.
set -u
cd /home/noorazman/dsh/strata/Strata
SHARD=/mnt/ssd/llm_models/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF/Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf
TOK="9419,11,821,803,369,7967,13,353,1044,264,11952,5617,303,220,17,15,17,21,13,10875,353,668,3184,488,883,821,1118,1834,13"
TAIL="--expert-cache auto --prefill 2048 --spec 4 --spec-min-p 0.5 --pool-workers 24 --mtp mtp/rt --max-context 8192 --kv fp16 --max-new 256 --tokens $TOK"
run() {
  echo mustoe8 | sudo -S -p '' bash -c "ulimit -l unlimited; cd /home/noorazman/dsh/strata/Strata && \
CUDA_VISIBLE_DEVICES=0 LD_LIBRARY_PATH=/usr/local/cuda/lib64 timeout 900 ./build-sm70/strata \
--pack packs/swift-iq3_xxs --native $SHARD --ple-gguf $SHARD --expert-profile data/expert-profile.bin \
$1" 2>&1
}
echo "=== E4a1: --gpu-stages ==="
run "$TAIL --gpu-stages" > /tmp/s17-e4-gpu-stages.log 2>&1
echo "rc=$?"
echo "=== E4a2: --gpu-only-full ==="
run "$TAIL --gpu-only-full" > /tmp/s17-e4-gpu-only.log 2>&1
echo "rc=$?"
echo "=== E4b: e2e A/B ==="
arms=(fast fast-attn fast-select fast fast-attn fast-select)
i=0
for a in "${arms[@]}"; do
  i=$((i+1))
  case "$a" in
    fast) label="s17e4-base-$i"; extra="";;
    fast-attn) label="s17e4-nofastattn-$i"; extra="--no-fast-attn";;
    fast-select) label="s17e4-nofastsel-$i"; extra="--no-fast-select";;
  esac
  echo "--- [$i/6] $label $extra ---"
  if [ -n "$extra" ]; then
    python3 bench/v100/bench.py "$label" --gpu 0 --workers 24 --max-new 256 --stats \
      --strata-flags "$extra"
  else
    python3 bench/v100/bench.py "$label" --gpu 0 --workers 24 --max-new 256 --stats
  fi
  echo "rc=$?"
  echo mustoe8 | sudo -S -p '' sh -c 'echo 3 > /proc/sys/vm/drop_caches' 2>/dev/null
  sleep 3
done
echo "E4 ALL DONE"
