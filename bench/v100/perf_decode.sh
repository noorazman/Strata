#!/bin/bash
# Capture a perf profile of the expert-pool decode window on one 256-token run.
set -u
cd /home/noorazman/dsh/strata/Strata || exit 1
LABEL=${1:-perf-decode}
WORKERS=${2:-28}
LOG="Logs/benchmarks/${LABEL}.engine.log"
DATA="Logs/cpu/${LABEL}.data"

rm -f "$LOG"
nohup bash -c "echo mustoe8 | sudo -S -p '' bash -c 'ulimit -l unlimited; cd /home/noorazman/dsh/strata/Strata && CUDA_VISIBLE_DEVICES=0 LD_LIBRARY_PATH=/usr/local/cuda/lib64 timeout 1500 ./build-sm70/strata --pack packs/swift-iq3_xxs --native /mnt/ssd/llm_models/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF/Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf --ple-gguf /mnt/ssd/llm_models/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF/Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf --expert-profile data/expert-profile.bin --expert-cache auto --prefill 2048 --spec 4 --spec-min-p 0.5 --pool-workers ${WORKERS} --mtp mtp/rt --max-context 8192 --kv fp16 --max-new 256 --stats --tokens 9419,11,821,803,369,7967,13,353,1044,264,11952,5617,303,220,17,15,17,21,13,10875,353,668,3184,488,883,821,1118,1834,13' > ${LOG} 2>&1" &

# wait for decode start (prefill line)
for i in $(seq 1 120); do
  grep -q "prefill 28 tokens in" "$LOG" 2>/dev/null && break
  sleep 0.5
done
PID=$(pgrep -x strata | head -1)
if [ -z "${PID:-}" ]; then echo "no strata pid"; exit 1; fi
echo "strata pid=$PID; recording 4 s"
sleep 1   # let the first round settle
echo mustoe8 | sudo -S -p '' perf record -F 99 -g -p "$PID" -o "$DATA" -- sleep 4 2>&1 | tail -2
echo "captured $(du -h "$DATA" | cut -f1)"
# let the run finish
for i in $(seq 1 60); do pgrep -x strata >/dev/null || break; sleep 1; done
grep -E "^decode|speculation" "$LOG" | head -3
