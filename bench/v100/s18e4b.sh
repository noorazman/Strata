#!/bin/bash
# Stage 1.8 — slow-arm re-runs + 32-token golden runs + int8 production A/B.
# (s18e4.sh base arms already completed 3 cycles; python3.10 argparse needs
# the --strata-flags=VALUE form for values that start with --.)
set -u
cd /home/noorazman/dsh/strata/Strata

run() { # label kv maxctx maxnew [strataflags]
  local label="$1" kv="$2" ctx="$3" new="$4" fl="$5"
  echo "=== $label ($kv ctx$ctx new$new ${fl:-none}) ==="
  if [ -n "$fl" ]; then
    python3 bench/v100/bench.py "$label" --gpu 0 --workers 24 --max-new "$new" \
      --kv "$kv" --max-context "$ctx" --stats --strata-flags="$fl"
  else
    python3 bench/v100/bench.py "$label" --gpu 0 --workers 24 --max-new "$new" \
      --kv "$kv" --max-context "$ctx" --stats
  fi
  echo "rc=$?"
  echo mustoe8 | sudo -S -p '' sh -c 'echo 3 > /proc/sys/vm/drop_caches' 2>/dev/null
  sleep 3
}

# 1) slow arms, 3 cycles interleaved (fp16 8192, 256 tok)
for c in 1 2 3; do
  run "s18e4-fp16-nofastattn-$c" fp16 8192 256 "--no-fast-attn"
  run "s18e4-fp16-nofastsel-$c" fp16 8192 256 "--no-fast-select"
done
# 2) 32-token golden prefix runs (fp16 8192)
run "s18e4-fp16-g32-base"       fp16 8192 32 ""
run "s18e4-fp16-g32-nofastattn" fp16 8192 32 "--no-fast-attn"
run "s18e4-fp16-g32-nofastsel"  fp16 8192 32 "--no-fast-select"
# 3) production config A/B (int8 32768, 256 tok) + int8 32-token prefixes
run "s18e4-int8-base"        int8 32768 256 ""
run "s18e4-int8-nofastattn"  int8 32768 256 "--no-fast-attn"
run "s18e4-int8-nofastsel"   int8 32768 256 "--no-fast-select"
run "s18e4-int8-g32-base"       int8 32768 32 ""
run "s18e4-int8-g32-nofastattn" int8 32768 32 "--no-fast-attn"
run "s18e4-int8-g32-nofastsel"  int8 32768 32 "--no-fast-select"
echo "S18E4B DONE"
