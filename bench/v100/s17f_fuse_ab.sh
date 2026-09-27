#!/bin/bash
# Stage 1.7 pre-work: historical ~52 tok/s fuse configuration investigation.
# 3x3 interleaved A/B on GPU0 (V100-32GB), 256-token decode, canonical tokens.
#   fused arm  : current Stage 1.6/1.7 defaults (pool mode 7 single-phase,
#                STRATA_POOL_ASYNC default ON, STRATA_POOL_PARK=2048 default,
#                arena pinned node0 default)
#   unfused arm: STRATA_POOL_UNFUSE=1 -> two-phase path (pre-Stage-1.4 behavior;
#                async dispatch auto-falls back to sync for the unfused path)
# drop_caches between arms (Stage 1.4 protocol). MCE ce_count bracketed.
set -u
cd /home/noorazman/dsh/strata/Strata
SUDO="echo mustoe8 | sudo -S -p ''"
mce() { cat /sys/devices/system/edac/mc/mc*/ce_count 2>/dev/null | paste -sd+ | bc; }
echo "MCE ce_count before: $(mce)"
arms=(base unf base unf base unf)
i=0
for a in "${arms[@]}"; do
  i=$((i+1))
  label="s17f-$a-$i"
  echo "=== [$i/6] $label ==="
  if [ "$a" = "unf" ]; then
    python3 bench/v100/bench.py "$label" --gpu 0 --workers 24 --max-new 256 --stats \
      --env STRATA_POOL_UNFUSE=1
  else
    python3 bench/v100/bench.py "$label" --gpu 0 --workers 24 --max-new 256 --stats
  fi
  rc=$?
  echo "=== [$i/6] $label rc=$rc ==="
  $SUDO sh -c 'echo 3 > /proc/sys/vm/drop_caches' 2>/dev/null
  sleep 3
done
echo "MCE ce_count after: $(mce)"
echo "ALL DONE"
