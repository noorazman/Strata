#!/bin/bash
# Stage 1.8 — nsys captures for the QSA attention-path analysis.
# 64-token decode window on the canonical prompt; no drop_caches (warm PLE
# from the previous run is fine: prefill is excluded from the decode window).
#   arms: base | nofastattn | nofastsel   (fp16 8192)
#         int8-base                        (production config)
set -u
cd /home/noorazman/dsh/strata/Strata
NSYS=/usr/local/cuda/bin/nsys
PACK="--pack packs/swift-iq3_xxs --native /mnt/ssd/llm_models/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF/Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf --ple-gguf /mnt/ssd/llm_models/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF/Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf --expert-profile data/expert-profile.bin"
FLAGS="${PACK} --kv fp16 --max-context 8192 --expert-cache auto --prefill 2048 --spec 4 --spec-min-p 0.5 --pool-workers 24 --mtp mtp/rt --tokens 9419,11,821,803,369,7967,13,353,1044,264,11952,5617,303,220,17,15,17,21,13,10875,353,668,3184,488,883,821,1118,1834,13 --max-new 64 --stats"
capture() { # label extra-flags
  local label="$1" extra="$2"
  echo "=== nsys $label ==="
  echo mustoe8 | sudo -S -p '' bash -c "
    ulimit -l unlimited
    cd /home/noorazman/dsh/strata/Strata
    export CUDA_VISIBLE_DEVICES=0 LD_LIBRARY_PATH=/usr/local/cuda/lib64
    $NSYS profile -o Logs/gpu/s18nsys-$label --force-overwrite true -t cuda \
      --cuda-graph-trace=node \
      ./build-sm70/strata $FLAGS $extra > Logs/gpu/s18nsys-$label.run.log 2>&1
  "
  echo "rc=$?"
  echo mustoe8 | sudo -S -p '' sh -c 'echo 3 > /proc/sys/vm/drop_caches' 2>/dev/null
  sleep 3
}
capture base ""
capture nofastattn "--no-fast-attn"
capture nofastsel "--no-fast-select"
capture int8-base "--kv int8 --max-context 32768"
for a in base nofastattn nofastsel int8-base; do
  $NSYS export --type sqlite --force-overwrite true \
    -o Logs/gpu/s18nsys-$a.sqlite Logs/gpu/s18nsys-$a.nsys-rep 2>/dev/null
done
echo "S18NSYS DONE"
