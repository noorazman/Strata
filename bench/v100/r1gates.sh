#!/bin/bash
# Release 1 — engine-level correctness gates (Gates 6-7), fresh engines, wide dequant ON.
# Per context: 32-tok golden-prefix run + 256-tok determinism x3 (fresh engine each),
# int8 KV, production flags (bench.py canonical), STRATA_MOE_DQ_WIDE=1.
set -u
cd /home/noorazman/dsh/strata/Strata
for C in 16384 32768; do
  for r in g32 det1 det2 det3; do
    if [ "$r" = g32 ]; then MN=32; else MN=256; fi
    echo "[$(date +%H:%M:%S)] ctx $C $r (max-new $MN)"
    python3 bench/v100/bench.py "r1g-$r-$C" --gpu 0 --workers 24 --max-new $MN --stats \
      --kv int8 --max-context $C --env STRATA_MOE_DQ_WIDE=1 > "Logs/gpu/r1g-$r-$C.out" 2>&1
    echo "[$(date +%H:%M:%S)] $r rc=$?"
  done
  echo "--- ctx $C determinism (det1 vs det2 vs det3, fresh engines)"
  python3 bench/v100/s18check.py "Logs/benchmarks/r1g-det1-$C.json" "Logs/benchmarks/r1g-det2-$C.json" "Logs/benchmarks/r1g-det3-$C.json"
  python3 bench/v100/s18check.py --diff "Logs/benchmarks/r1g-det1-$C.json" "Logs/benchmarks/r1g-det2-$C.json"
  python3 bench/v100/s18check.py --diff "Logs/benchmarks/r1g-det1-$C.json" "Logs/benchmarks/r1g-det3-$C.json"
  echo "--- ctx $C golden prefix"
  python3 bench/v100/s18check.py --prefix "Logs/benchmarks/r1g-g32-$C.json" 32
  grep -hE "first-|decode [0-9]+ tokens" "Logs/gpu/r1g-g32-$C.out" "Logs/gpu/r1g-det1-$C.out" 2>/dev/null | head -6
done
echo "[$(date +%H:%M:%S)] ALL RELEASE-1 ENGINE GATES DONE"
