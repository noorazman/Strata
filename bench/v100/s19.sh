#!/bin/bash
# Stage 1.9 — wait_flag_ge A/B/C investigation (profile-only, no engine changes).
#
# Arms:
#   s19-g32       32-tok golden prefix check (no nsys)
#   s19-det-1/2   256-tok determinism pair (no nsys); both must be byte-identical
#                 to the Stage 1.7/1.8 fp16 golden md5 c1517d02473fbc06b5cf415ea1f8be63
#   s19-hostdiag  256-tok with the opt-in host diagnostics (STRATA_WAIT_ITERS /
#                 STRATA_SFENCE_DIAG / STRATA_HEAD_DIAG / STRATA_POOL_ASYNC_DIAG),
#                 no nsys: the host-side view of doorbell visibility, plan/A/B/C
#                 publish latency, pool async phases, and the round-end host path
#   s19-base      nsys 256-tok fp16/8192 node trace, STRATA_WAIT_ITERS=1
#                 (same workload as the s17x-base old-binary capture)
#   s19-int8      nsys 256-tok int8/32768 node trace (the production service config)
#
# Prerequisite: strata.service stopped (GPU0 free).  Do NOT restart it afterwards.
set -u
cd /home/noorazman/dsh/strata/Strata
NSYS=/usr/local/cuda/bin/nsys
SHARD=/mnt/ssd/llm_models/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF/Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf
TOK="9419,11,821,803,369,7967,13,353,1044,264,11952,5617,303,220,17,15,17,21,13,10875,353,668,3184,488,883,821,1118,1834,13"
PACK="--pack packs/swift-iq3_xxs --native $SHARD --ple-gguf $SHARD --expert-profile data/expert-profile.bin"

echo "[$(date +%H:%M:%S)] service state: $(systemctl is-active strata.service)"
echo "[$(date +%H:%M:%S)] GPU0: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader -i 0)"

if [ "${1:-}" != "--skip-sanity" ]; then
echo "[$(date +%H:%M:%S)] PHASE 1: sanity arms"
python3 bench/v100/bench.py s19-g32 --gpu 0 --workers 24 --max-new 32 --stats > Logs/gpu/s19-g32.out 2>&1
echo "g32 rc=$?"
python3 bench/v100/bench.py s19-det-1 --gpu 0 --workers 24 --max-new 256 --stats > Logs/gpu/s19-det-1.out 2>&1
echo "det1 rc=$?"
python3 bench/v100/bench.py s19-det-2 --gpu 0 --workers 24 --max-new 256 --stats > Logs/gpu/s19-det-2.out 2>&1
echo "det2 rc=$?"
echo "golden/determinism check:"
python3 bench/v100/s18check.py Logs/benchmarks/s19-g32.json 2>/dev/null | head -3
python3 bench/v100/s18check.py Logs/benchmarks/s19-det-1.json Logs/benchmarks/s19-det-2.json 2>/dev/null | head -6
python3 bench/v100/s18check.py --prefix Logs/benchmarks/s19-g32.json 32 2>/dev/null
python3 bench/v100/bench.py s19-hostdiag --gpu 0 --workers 24 --max-new 256 --stats \
  --env STRATA_WAIT_ITERS=1 --env STRATA_SFENCE_DIAG=1 --env STRATA_HEAD_DIAG=1 \
  --env STRATA_POOL_ASYNC_DIAG=1 > Logs/gpu/s19-hostdiag.out 2>&1
echo "hostdiag rc=$?"
fi

echo "[$(date +%H:%M:%S)] PHASE 2: nsys captures"
cap() { # label kv ctx
  local label="$1" kv="$2" ctx="$3"
  echo "[$(date +%H:%M:%S)] CAP s19-$label ($kv $ctx)"
  echo mustoe8 | sudo -S -p '' bash -c "
    ulimit -l unlimited
    cd /home/noorazman/dsh/strata/Strata
    export CUDA_VISIBLE_DEVICES=0 LD_LIBRARY_PATH=/usr/local/cuda/lib64
    export STRATA_WAIT_ITERS=1
    $NSYS profile -o Logs/gpu/s19-$label --force-overwrite true -t cuda \
      --cuda-graph-trace=node \
      ./build-sm70/strata $PACK --kv $kv --max-context $ctx --expert-cache auto \
      --prefill 2048 --spec 4 --spec-min-p 0.5 --pool-workers 24 --mtp mtp/rt \
      --max-new 256 --tokens $TOK > Logs/gpu/s19-$label.run.log 2>&1
  "
  echo "s19-$label rc=$?"
  $NSYS export --type sqlite --force-overwrite true -o Logs/gpu/s19-$label.sqlite Logs/gpu/s19-$label.nsys-rep 2>/dev/null
  echo "export rc=$?"
  echo mustoe8 | sudo -S -p '' sh -c 'echo 3 > /proc/sys/vm/drop_caches' 2>/dev/null
  sleep 3
}
cap base fp16 8192
cap int8 int8 32768
echo "S19 CAMPAIGN DONE $(date +%H:%M:%S)"
