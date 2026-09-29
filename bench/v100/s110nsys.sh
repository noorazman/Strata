#!/bin/bash
# Stage 1.10 — nsys captures for the flag-A phase attribution (prefill vs first-token vs decode).
#
# Arms (all on GPU0, strata.service stopped, ulimit -l unlimited for the 40 GiB pinned arena):
#   s110nsys-16k-off  engine --serve int8/32768, 16K-token corpus prompt + 64 max-new via stdin,
#                     STRATA_WAIT_ITERS=1 STRATA_TTFT=1, nsys cuda graph-node trace.  FENCE OFF.
#   s110nsys-16k-on   same with STRATA_WAIT_FENCE=1 (the opt-in periodic __threadfence_system)
#   s110nsys-256-fence  the Stage 1.9 workload (29-tok canonical prompt, 256 max-new, fp16/8192)
#                     with STRATA_WAIT_FENCE=1 -> round-head A distribution with the fence,
#                     comparable against the existing Logs/gpu/s19-base.sqlite (fence off).
#
# The serve-mode captures feed the engine exactly one request (GEN line + QUIT), so the trace
# covers: load -> prefill (16K) -> first verify window (first token) -> 64 decode tokens.
set -u
cd /home/noorazman/dsh/strata/Strata
NSYS=/usr/local/cuda/bin/nsys
SHARD=/mnt/ssd/llm_models/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF/Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf
PACK="--pack packs/swift-iq3_xxs --native $SHARD --ple-gguf $SHARD --expert-profile data/expert-profile.bin"
TOK="9419,11,821,803,369,7967,13,353,1044,264,11952,5617,303,220,17,15,17,21,13,10875,353,668,3184,488,883,821,1118,1834,13"
SERVE_FLAGS="--prefill 2048 --spec 4 --spec-min-p 0.5 --pool-workers 24 --mtp mtp/rt --expert-cache auto"

# the 16K corpus prefix as a GEN line (16384 prompt tokens + 64 max-new)
python3 - <<'EOF'
ids = open("bench/v100/corpus-265k.ids").read().split()[:16384]
open("Logs/gpu/s110nsys-16k-gen.txt", "w").write(f"GEN 64 {','.join(ids)}\nQUIT\n")
print("wrote Logs/gpu/s110nsys-16k-gen.txt (16384 prompt tokens)")
EOF

cap_serve() { # label fence_env
  local label="$1" fence="$2"
  echo "[$(date +%H:%M:%S)] CAP $label (fence=${fence:-off})"
  echo mustoe8 | sudo -S -p '' bash -c "
    ulimit -l unlimited
    cd /home/noorazman/dsh/strata/Strata
    export CUDA_VISIBLE_DEVICES=0 LD_LIBRARY_PATH=/usr/local/cuda/lib64
    export STRATA_WAIT_ITERS=1 STRATA_TTFT=1 ${fence}
    $NSYS profile -o Logs/gpu/$label --force-overwrite true -t cuda \
      --cuda-event-trace=false --cuda-graph-trace=node \
      ./build-sm70/strata $PACK --kv int8 --max-context 32768 $SERVE_FLAGS --serve \
      < Logs/gpu/s110nsys-16k-gen.txt > Logs/gpu/$label.run.log 2>&1
  "
  echo "$label rc=$?"
  $NSYS export --type sqlite --force-overwrite true -o Logs/gpu/$label.sqlite Logs/gpu/$label.nsys-rep 2>/dev/null
  echo "export rc=$?"
  sleep 3
}

case "${1:-}" in
  16k) cap_serve s110nsys-16k-off "" ; cap_serve s110nsys-16k-on "STRATA_WAIT_FENCE=1" ;;
  256)
    echo "[$(date +%H:%M:%S)] CAP s110nsys-256-fence (fp16/8192, 256 tok, STRATA_WAIT_FENCE=1)"
    echo mustoe8 | sudo -S -p '' bash -c "
      ulimit -l unlimited
      cd /home/noorazman/dsh/strata/Strata
      export CUDA_VISIBLE_DEVICES=0 LD_LIBRARY_PATH=/usr/local/cuda/lib64
      export STRATA_WAIT_FENCE=1
      $NSYS profile -o Logs/gpu/s110nsys-256-fence --force-overwrite true -t cuda \
        --cuda-event-trace=false --cuda-graph-trace=node \
        ./build-sm70/strata $PACK --kv fp16 --max-context 8192 --expert-cache auto $SERVE_FLAGS \
        --max-new 256 --tokens $TOK > Logs/gpu/s110nsys-256-fence.run.log 2>&1
    "
    echo "256-fence rc=$?"
    $NSYS export --type sqlite --force-overwrite true -o Logs/gpu/s110nsys-256-fence.sqlite \
      Logs/gpu/s110nsys-256-fence.nsys-rep 2>/dev/null
    echo "export rc=$?"
    ;;
  *) sed -n '2,22p' "$0"; exit 1 ;;
esac
echo "S110 NSYS DONE $(date +%H:%M:%S)"
