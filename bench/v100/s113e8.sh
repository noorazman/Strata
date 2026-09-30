#!/bin/bash
# Stage 1.13 E8 A/B: expert-resident H2D byte reduction - grow the resident expert cache from the
# production 8,000-slot profile to ~14,000 slots on the new top-14,000 profile (data/expert-profile-14k.bin).
#
#   s113e8.sh gate <C>     engine-level gate at C with the 14K profile (32-tok golden prefix +
#                          256-tok determinism x2, int8 KV, production flags) + decode rate
#   s113e8.sh ctx <C>      serve-path TTFT/prefill/decode point at max-context C with the 14K
#                          profile (s110.sh ctx pattern; config carries --expert-cache 14000 +
#                          the 14K profile instead of --expert-cache auto + the 8K profile)
#
# The 8,000-slot (production) side is the fresh `s110.sh ctx <C>` baseline (same build).
# No engine code change: E8 is the existing --expert-cache N mechanism + a larger profile.
set -u
cd /home/noorazman/dsh/strata/Strata
REPO=$(pwd)
CLIENT="python3 bench/v100/s110_client.py"
SLOTS="${SLOTS:-14000}"
PROFILE="data/expert-profile-14k.bin"

die() { echo "[$(date +%H:%M:%S)] FATAL: $*"; exit 1; }
pgrep -x strata >/dev/null && die "a strata engine is already running"
mib=$(nvidia-smi -i 0 --query-gpu=memory.used --format=csv,noheader,nounits)
[ "$mib" -gt 2048 ] && die "GPU0 not idle (used $mib MiB)"
[ "$(systemctl is-active strata.service 2>/dev/null)" = "active" ] && die "strata.service is active"
[ -f "$PROFILE" ] || die "$PROFILE missing - build it with bench/v100/e8_profile.py first"

start_server() { # $1 = config json
  local cfg="$1"
  local tag="s113-$(basename "$cfg" .json)"
  CUDA_VISIBLE_DEVICES=0 LD_LIBRARY_PATH=/usr/local/cuda/lib64 STRATA_TTFT=1 env ${EXTRA_ENV:-} \
    nohup .venv/bin/python -m serve.server --engine strata --config "$cfg" --host 127.0.0.1 --port 8180 \
    > "Logs/gpu/${tag}.serve.log" 2>&1 &
  local SERVER_PID=$!
  echo "$SERVER_PID" > /tmp/s113e8-server.pid
  local i
  for i in $(seq 1 150); do
    if grep -q "ready: http" "Logs/gpu/${tag}.serve.log" 2>/dev/null; then
      echo "[$(date +%H:%M:%S)] server ready ($tag, pid $SERVER_PID)"; return 0
    fi
    kill -0 "$SERVER_PID" 2>/dev/null || { echo "[$(date +%H:%M:%S)] server died:"; tail -25 "Logs/gpu/${tag}.serve.log"; return 1; }
    sleep 2
  done
  echo "[$(date +%H:%M:%S)] server start timed out"; return 1
}
stop_server() {
  local p
  [ -f /tmp/s113e8-server.pid ] || return 0
  p=$(cat /tmp/s113e8-server.pid)
  kill -INT "$p" 2>/dev/null
  local i
  for i in $(seq 1 45); do kill -0 "$p" 2>/dev/null || break; sleep 2; done
  kill -0 "$p" 2>/dev/null && { echo "[$(date +%H:%M:%S)] forcing server"; kill -KILL "$p" 2>/dev/null; sleep 2; }
  wait "$p" 2>/dev/null
  rm -f /tmp/s113e8-server.pid
  for i in $(seq 1 30); do pgrep -x strata >/dev/null || break; sleep 2; done
  if pgrep -x strata >/dev/null; then echo "[$(date +%H:%M:%S)] WARNING: strata engine still running"; fi
  echo "[$(date +%H:%M:%S)] server stopped"
}
vram_sampler() { # $1 = csv out
  ( local t=0
    while :; do
      nvidia-smi -i 0 --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits
      sleep 0.5; t=$((t + 1))
    done ) > "$1" 2>/dev/null &
  echo $! > /tmp/s113e8-vram.pid
}
stop_sampler() {
  [ -f /tmp/s113e8-vram.pid ] || return 0
  kill "$(cat /tmp/s113e8-vram.pid)" 2>/dev/null; wait "$(cat /tmp/s113e8-vram.pid)" 2>/dev/null
  rm -f /tmp/s113e8-vram.pid
}
meminfo() {
  awk '/MemAvailable/{a=$2} /^MemFree:/{f=$2} END{print "avail=" a " free=" f}' /proc/meminfo
}
overhead_of() {
  python3 - "$1" <<'EOF'
import json, sys
label = sys.argv[1]
val = None
with open("/home/noorazman/dsh/strata/Strata/Logs/benchmarks/s110-client.jsonl") as f:
    for line in f:
        d = json.loads(line)
        if d.get("label") == label:
            val = d.get("template_overhead")
print(val if val is not None else -1)
EOF
}

mkctx_cfg_e8() { # $1 = max-context -> bench/v100/s113e8-ctx-$1.json (production config + E8 args)
  local c="$1" out="bench/v100/s113e8-ctx-${c}.json"
  python3 - "$c" "$out" "$SLOTS" "$PROFILE" <<'EOF'
import json, sys
c, out, slots, prof = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
cfg = json.load(open("/home/noorazman/dsh/strata/Strata/strata-swift-iq3_xxs.json"))
args = []
skip = False
for a in cfg["args"]:
    if skip: skip = False; continue
    if a == "--max-context": skip = True; continue
    if a == "--expert-cache": skip = True; args += ["--expert-cache", slots]; continue
    if a == "--expert-profile": skip = True; args += ["--expert-profile", prof]; continue
    args.append(a)
args += ["--max-context", c]
cfg["args"] = args
cfg["log"] = f"/home/noorazman/dsh/strata/Strata/Logs/benchmarks/s113e8-ctx-{c}-engine.log"
cfg["model_name"] = f"swift-iq3_xxs-e8-ctx{c}"
json.dump(cfg, open(out, "w"), indent=1)
EOF
}

case "${1:-}" in
  gate)
    c="${2:-16384}"
    echo "[$(date +%H:%M:%S)] E8 gate ctx=$c ($SLOTS slots, $PROFILE)"
    python3 bench/v100/bench.py "s113e8-g${c}-g32" --gpu 0 --workers 24 --max-new 32 --stats \
      --kv int8 --max-context "$c" \
      --strata-flags "--expert-cache $SLOTS --expert-profile $PROFILE" > "Logs/gpu/s113e8-g${c}-g32.out" 2>&1
    echo "g32 rc=$?"
    python3 bench/v100/bench.py "s113e8-g${c}-det1" --gpu 0 --workers 24 --max-new 256 --stats \
      --kv int8 --max-context "$c" \
      --strata-flags "--expert-cache $SLOTS --expert-profile $PROFILE" > "Logs/gpu/s113e8-g${c}-det1.out" 2>&1
    echo "det1 rc=$?"
    python3 bench/v100/bench.py "s113e8-g${c}-det2" --gpu 0 --workers 24 --max-new 256 --stats \
      --kv int8 --max-context "$c" \
      --strata-flags "--expert-cache $SLOTS --expert-profile $PROFILE" > "Logs/gpu/s113e8-g${c}-det2.out" 2>&1
    echo "det2 rc=$?"
    python3 bench/v100/s18check.py "Logs/benchmarks/s113e8-g${c}-det1.json" "Logs/benchmarks/s113e8-g${c}-det2.json"
    # E8 changes the residency set (8000 -> 14000 resident experts), so the numerics differ from the
    # old 8000-slot golden. The E8-internal golden check: the 32-tok run must be the exact prefix of
    # the 256-tok run (same config, deterministic), and det1/det2 must be byte-identical.
    python3 bench/v100/s18check.py --diff "Logs/benchmarks/s113e8-g${c}-g32.json" "Logs/benchmarks/s113e8-g${c}-det1.json"
    python3 bench/v100/s18check.py --prefix "Logs/benchmarks/s113e8-g${c}-g32.json" 32
    grep -hE "R4 expert-cache hits|expert cache|pre-filled|borrow" "Logs/gpu/s113e8-g${c}-det1.out" | head -6
    python3 - "$c" <<'EOF'
import json, sys
c = sys.argv[1]
d = json.load(open(f"/home/noorazman/dsh/strata/Strata/Logs/benchmarks/s113e8-g{c}-det1.json"))
for k in ("prefill_tps", "decode_tps", "spec_accept", "spec_tpr", "mtp_draft_ms_round", "gpu_only_full_ms"):
    print(f"{k}: {d.get(k)}")
EOF
    echo "[$(date +%H:%M:%S)] E8 GATE DONE"
    ;;
  ctx)
    c="${2:-}"
    [ -n "$c" ] || die "usage: s113e8.sh ctx <max-context>"
    mkctx_cfg_e8 "$c" || die "config gen failed"
    cfg="bench/v100/s113e8-ctx-${c}.json"
    tag="s113e8-ctx-${c}"
    rm -f "Logs/benchmarks/${tag}-engine.log"
    start_server "$cfg" || die "server start failed (e8 ctx $c)"
    oh="" n=""
    echo "[$(date +%H:%M:%S)] e8 ctx $c: overhead probe"
    $CLIENT --port 8180 --prompt-tokens 100 --max-new 8 --thinking on --label "E8-${c}-oh" || die "oh"
    oh=$(overhead_of "E8-${c}-oh")
    [ "$oh" -lt 0 ] && die "overhead probe failed ($oh)"
    echo "[$(date +%H:%M:%S)] e8 ctx $c: warmup 1024"
    $CLIENT --port 8180 --prompt-tokens 1024 --max-new 16 --thinking on --label "E8-${c}-warm" || die "warm"
    n=$((c - oh - 256 - 8))
    echo "[$(date +%H:%M:%S)] e8 ctx $c: main run 1 (fill prompt $n, 256 max-new) - $(meminfo)"
    vram_sampler "Logs/benchmarks/${tag}-vram1.csv"
    $CLIENT --port 8180 --prompt-tokens "$n" --max-new 256 --thinking on --label "E8-${c}-r1" --timeout 3600 || die "E8 r1"
    stop_sampler
    echo "[$(date +%H:%M:%S)] e8 ctx $c: main run 2 (warm) - $(meminfo)"
    vram_sampler "Logs/benchmarks/${tag}-vram2.csv"
    $CLIENT --port 8180 --prompt-tokens "$n" --max-new 256 --thinking on --label "E8-${c}-r2" --timeout 3600 || die "E8 r2"
    stop_sampler
    stop_server
    grep "strata serve ttft" "Logs/benchmarks/${tag}-engine.log" 2>/dev/null > "Logs/benchmarks/${tag}-engine-ttft.jsonl"
    python3 - "$c" <<'EOF'
import json, sys
c = sys.argv[1]
a = b = None
with open("/home/noorazman/dsh/strata/Strata/Logs/benchmarks/s110-client.jsonl") as f:
    for line in f:
        d = json.loads(line)
        if d.get("label") == f"E8-{c}-r1": a = d
        if d.get("label") == f"E8-{c}-r2": b = d
if a and b:
    print(f"[e8 ctx {c}] fill-prompt r1 md5={a['text_md5']} r2 md5={b['text_md5']} "
          f"{'IDENTICAL' if a['text_md5'] == b['text_md5'] else 'DIVERGED (normal serve-mode)'}")
    print(f"[e8 ctx {c}] r1 TTFT {a['t_first_reasoning_ms']/1000:.2f} s, decode {a.get('steady_decode_tps')} tok/s")
    print(f"[e8 ctx {c}] r2 TTFT {b['t_first_reasoning_ms']/1000:.2f} s, decode {b.get('steady_decode_tps')} tok/s")
else:
    print(f"[e8 ctx {c}] determinism check: missing runs (r1={bool(a)} r2={bool(b)})")
EOF
    echo "[$(date +%H:%M:%S)] E8 CTX $c DONE"
    ;;
  *)
    sed -n '2,13p' "$0"; exit 1
    ;;
esac
