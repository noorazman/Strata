#!/bin/bash
# Stage 1.10 — long-context sweep driver: for each max-context C, run the server-based
# fill-prompt measurements (s110.sh ctx C) and the engine-level correctness gates
# (s110.sh ctxgate C).  Ascending order so the riskiest (262K) lands last; if it OOMs,
# the largest safe size below it is already measured.
#
#   s110_sweep.sh [C1 [C2 ...]]   (default: 4096 8192 16384 32768 65536 131072 262144)
set -u
cd /home/noorazman/dsh/strata/Strata
if [ $# -eq 0 ]; then
  set -- 4096 8192 16384 32768 65536 131072 262144
fi
for c in "$@"; do
  echo "==================== CONTEXT $c ===================="
  bash bench/v100/s110.sh ctx "$c" || echo "!! ctx $c FAILED (continuing)"
  bash bench/v100/s110.sh ctxgate "$c" || echo "!! ctxgate $c FAILED (continuing)"
done
echo "S110 SWEEP DONE $(date +%H:%M:%S)"
