#!/bin/bash
# Stage 1.10 — Prefill / TTFT / Long-Context campaign (32 GB V100 GPU0, strata.service STOPPED during runs).
#
# Subcommands:
#   s110.sh ttft          Phase A: TTFT matrix on the production 32K config (thinking on/off;
#                         4K/8K/16K/32K prompts) via the live serve path (STRATA_TTFT=1)
#   s110.sh ctx <C>       Phase B: one long-context point (max-context C, int8 KV): overhead probe,
#                         1K warmup, then the fill-prompt main run TWICE (run1 cold / run2 warm),
#                         256 max-new each; VRAM sampled at 2 Hz
#   s110.sh ctxgate <C>   Phase B2: engine-level correctness gate for context C (32-tok golden prefix
#                         + 256-tok determinism x2, int8 KV, production flags)
#   s110.sh fence         Phase F: STRATA_WAIT_FENCE e2e A/B on the Stage 1.9 workload (fp16/8192)
#                         and the production config (int8/32768): 256-tok decode rate + determinism
#                         with/without the opt-in fence
#
# Client results:  Logs/benchmarks/s110-client.jsonl  (one JSON line per request, label field)
# Server logs:     Logs/gpu/s110-<tag>.serve.log      (python serve stderr: "strata serve ttft:" lines)
# Engine logs:     the config's "log" path            (engine stderr: "strata serve ttft:" + prefill stats)
set -u
cd /home/noorazman/dsh/strata/Strata
REPO=$(pwd)
CLIENT="python3 bench/v100/s110_client.py"
ENGINE_LOG_PROD="$REPO/strata-swift-iq3_xxs.log"
SERVER_PID=""
VRAM_PID=""

die() { echo "[$(date +%H:%M:%S)] FATAL: $*"; exit 1; }

start_server() { # $1 = config json
  local cfg="$1"
  local tag="s110-$(basename "$cfg" .json)"
  # Stage 1.11: EXTRA_ENV (e.g. STRATA_MTP_OVERLAP=1) reaches the engine through the serve process
  CUDA_VISIBLE_DEVICES=0 LD_LIBRARY_PATH=/usr/local/cuda/lib64 STRATA_TTFT=1 env ${EXTRA_ENV:-} \
    nohup .venv/bin/python -m serve.server --engine strata --config "$cfg" --host 127.0.0.1 --port 8180 \
    > "Logs/gpu/${tag}.serve.log" 2>&1 &
  SERVER_PID=$!
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
  [ -n "$SERVER_PID" ] || return 0
  kill -INT "$SERVER_PID" 2>/dev/null
  local i
  for i in $(seq 1 45); do kill -0 "$SERVER_PID" 2>/dev/null || break; sleep 2; done
  kill -0 "$SERVER_PID" 2>/dev/null && { echo "[$(date +%H:%M:%S)] forcing server"; kill -KILL "$SERVER_PID" 2>/dev/null; sleep 2; }
  wait "$SERVER_PID" 2>/dev/null
  for i in $(seq 1 30); do pgrep -x strata >/dev/null || break; sleep 2; done
  if pgrep -x strata >/dev/null; then echo "[$(date +%H:%M:%S)] WARNING: strata engine still running"; fi
  SERVER_PID=""
  echo "[$(date +%H:%M:%S)] server stopped"
}

vram_sampler() { # $1 = csv out; $2 optional max seconds
  local out="$1" max="${2:-0}"
  ( local t=0
    while :; do
      nvidia-smi -i 0 --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits
      sleep 0.5; t=$((t + 1))
      [ "$max" -gt 0 ] && [ "$t" -ge $((max * 2)) ] && break
    done ) > "$out" 2>/dev/null &
  VRAM_PID=$!
}
stop_sampler() {
  [ -n "$VRAM_PID" ] || return 0
  kill "$VRAM_PID" 2>/dev/null; wait "$VRAM_PID" 2>/dev/null; VRAM_PID=""
}

meminfo() { # prints "avail=.. free=.."
  awk '/MemAvailable/{a=$2} /^MemFree:/{f=$2} END{print "avail=" a " free=" f}' /proc/meminfo
}

overhead_of() { # label -> template overhead from the last jsonl line with that label
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

require_idle() {
  pgrep -x strata >/dev/null && die "a strata engine is already running"
  local mib; mib=$(nvidia-smi -i 0 --query-gpu=memory.used --format=csv,noheader,nounits)
  [ "$mib" -gt 2048 ] && die "GPU0 not idle (used $mib MiB)"
  [ "$(systemctl is-active strata.service 2>/dev/null)" = "active" ] && die "strata.service is active (it must be stopped during profiling)"
}

mkctx_cfg() { # $1 = max-context -> writes bench/v100/s110-ctx-$1.json
  local c="$1" out="bench/v100/s110-ctx-${c}.json"
  python3 - "$c" "$out" <<'EOF'
import json, sys
c, out = sys.argv[1], sys.argv[2]
cfg = json.load(open("/home/noorazman/dsh/strata/Strata/strata-swift-iq3_xxs.json"))
args = []
skip = False
for a in cfg["args"]:
    if skip: skip = False; continue
    if a == "--max-context": skip = True; continue
    args.append(a)
args += ["--max-context", c]
cfg["args"] = args
cfg["log"] = f"/home/noorazman/dsh/strata/Strata/Logs/benchmarks/s110-ctx-{c}-engine.log"
cfg["model_name"] = f"swift-iq3_xxs-ctx{c}"
json.dump(cfg, open(out, "w"), indent=1)
EOF
}

phase_ttft() {
  require_idle
  local L1; L1=$(wc -c < "$ENGINE_LOG_PROD")
  start_server "$REPO/strata-swift-iq3_xxs.json" || die "server start failed"
  local oh_on oh_off n
  echo "[$(date +%H:%M:%S)] overhead probes"
  $CLIENT --port 8180 --prompt-tokens 100 --max-new 8 --thinking on  --label A-oh-on  || die "oh-on"
  $CLIENT --port 8180 --prompt-tokens 100 --max-new 8 --thinking off --label A-oh-off || die "oh-off"
  oh_on=$(overhead_of A-oh-on);  oh_off=$(overhead_of A-oh-off)
  [ "$oh_on" -lt 0 ] || [ "$oh_off" -lt 0 ] && die "template overhead probe failed ($oh_on/$oh_off)"
  echo "[$(date +%H:%M:%S)] template overhead: thinking-on=$oh_on thinking-off=$oh_off"
  for n in 4096 8192 16384; do
    for th in on off; do
      echo "[$(date +%H:%M:$((n % 60)))] === A $n tok thinking=$th"
      $CLIENT --port 8180 --prompt-tokens "$n" --max-new 128 --thinking "$th" --label "A-${n}-$th" || die "A-$n-$th"
    done
  done
  local n32
  for th in on off; do
    if [ "$th" = on ]; then n32=$((32768 - oh_on - 128 - 8)); else n32=$((32768 - oh_off - 128 - 8)); fi
    echo "[$(date +%H:%M:%S)] === A 32K fill ($n32 tok) thinking=$th"
    $CLIENT --port 8180 --prompt-tokens "$n32" --max-new 128 --thinking "$th" --label "A-32K-$th" || die "A-32K-$th"
  done
  stop_server
  # engine-side stages for this phase (new log bytes only)
  tail -c +$((L1 + 1)) "$ENGINE_LOG_PROD" | grep "strata serve ttft" > "Logs/benchmarks/s110-A-engine-ttft.jsonl"
  echo "[$(date +%H:%M:%S)] PHASE A DONE"
}

phase_ctx() { # $1 = max-context
  local c="$1"
  require_idle
  [ "$c" -ge 4096 ] && [ "$c" -le 262144 ] || die "context $c out of range"
  mkctx_cfg "$c" || die "config gen failed"
  local cfg="bench/v100/s110-ctx-${c}.json"
  local tag="s110-ctx-${c}"
  rm -f "Logs/benchmarks/${tag}-engine.log"
  start_server "$cfg" || die "server start failed (ctx $c)"
  local oh n
  echo "[$(date +%H:%M:%S)] ctx $c: overhead probe"
  $CLIENT --port 8180 --prompt-tokens 100 --max-new 8 --thinking on --label "B-${c}-oh" || die "B-oh"
  oh=$(overhead_of "B-${c}-oh")
  [ "$oh" -lt 0 ] && die "overhead probe failed"
  echo "[$(date +%H:%M:%S)] ctx $c: warmup 1024"
  $CLIENT --port 8180 --prompt-tokens 1024 --max-new 16 --thinking on --label "B-${c}-warm" || die "B-warm"
  n=$((c - oh - 256 - 8))
  echo "[$(date +%H:%M:%S)] ctx $c: main run 1 (fill prompt $n, 256 max-new) - $(meminfo)"
  vram_sampler "Logs/benchmarks/${tag}-vram1.csv"
  $CLIENT --port 8180 --prompt-tokens "$n" --max-new 256 --thinking on --label "B-${c}-r1" --timeout 3600 || die "B-r1"
  stop_sampler
  echo "[$(date +%H:%M:%S)] ctx $c: main run 2 (warm) - $(meminfo)"
  vram_sampler "Logs/benchmarks/${tag}-vram2.csv"
  $CLIENT --port 8180 --prompt-tokens "$n" --max-new 256 --thinking on --label "B-${c}-r2" --timeout 3600 || die "B-r2"
  stop_sampler
  stop_server
  # engine-side stages for this context
  grep "strata serve ttft" "Logs/benchmarks/${tag}-engine.log" 2>/dev/null > "Logs/benchmarks/${tag}-engine-ttft.jsonl"
  # determinism of the two fill-prompt runs (visible text digest)
  python3 - "$c" <<'EOF'
import json, sys
c = sys.argv[1]
a = b = None
with open("/home/noorazman/dsh/strata/Strata/Logs/benchmarks/s110-client.jsonl") as f:
    for line in f:
        d = json.loads(line)
        if d.get("label") == f"B-{c}-r1": a = d
        if d.get("label") == f"B-{c}-r2": b = d
if a and b:
    print(f"[ctx {c}] fill-prompt determinism: r1 md5={a['text_md5']} r2 md5={b['text_md5']} "
          f"{'IDENTICAL' if a['text_md5'] == b['text_md5'] else 'DIVERGED'}")
else:
    print(f"[ctx {c}] determinism check: missing runs (r1={bool(a)} r2={bool(b)})")
EOF
  echo "[$(date +%H:%M:%S)] PHASE B CTX $c DONE"
}

phase_ctxgate() { # $1 = max-context
  local c="$1"
  require_idle
  local base="s110-ctx${c}"
  echo "[$(date +%H:%M:%S)] ctx $c gate: 32-tok golden (int8)"
  python3 bench/v100/bench.py "${base}-g32" --gpu 0 --workers 24 --max-new 32 --stats \
    --kv int8 --max-context "$c" > "Logs/gpu/${base}-g32.out" 2>&1
  echo "g32 rc=$?"
  echo "[$(date +%H:%M:%S)] ctx $c gate: 256-tok determinism x2 (int8)"
  python3 bench/v100/bench.py "${base}-det1" --gpu 0 --workers 24 --max-new 256 --stats \
    --kv int8 --max-context "$c" > "Logs/gpu/${base}-det1.out" 2>&1
  echo "det1 rc=$?"
  python3 bench/v100/bench.py "${base}-det2" --gpu 0 --workers 24 --max-new 256 --stats \
    --kv int8 --max-context "$c" > "Logs/gpu/${base}-det2.out" 2>&1
  echo "det2 rc=$?"
  python3 bench/v100/s18check.py "Logs/benchmarks/${base}-det1.json" "Logs/benchmarks/${base}-det2.json"
  grep -h "first-" "Logs/gpu/${base}-g32.out" 2>/dev/null
  python3 bench/v100/s18check.py --prefix "Logs/benchmarks/${base}-g32.json" 32
  echo "[$(date +%H:%M:%S)] PHASE B2 CTX $c GATE DONE"
}

phase_fence() {
  require_idle
  local TOK="9419,11,821,803,369,7967,13,353,1044,264,11952,5617,303,220,17,15,17,21,13,10875,353,668,3184,488,883,821,1118,1834,13"
  # The Stage 1.9 workload: 29-tok canonical prompt, 256 max-new.  fp16/8192 = the frozen reference config;
  # int8/32768 = the production service config.  Fence OFF vs ON (opt-in STRATA_WAIT_FENCE=1).
  for kv_ctx in "fp16 8192" "int8 32768"; do
    local kv ctx; kv="${kv_ctx% *}"; ctx="${kv_ctx#* }"
    local tag="s110-f-${kv}-${ctx}"
    echo "[$(date +%H:%M:%S)] fence A/B: $kv/$ctx OFF"
    python3 bench/v100/bench.py "${tag}-off" --gpu 0 --workers 24 --max-new 256 --stats \
      --kv "$kv" --max-context "$ctx" > "Logs/gpu/${tag}-off.out" 2>&1
    echo "off rc=$?"
    echo "[$(date +%H:%M:%S)] fence A/B: $kv/$ctx ON (STRATA_WAIT_FENCE=1)"
    python3 bench/v100/bench.py "${tag}-on" --gpu 0 --workers 24 --max-new 256 --stats \
      --kv "$kv" --max-context "$ctx" --env STRATA_WAIT_FENCE=1 > "Logs/gpu/${tag}-on.out" 2>&1
    echo "on rc=$?"
  done
  python3 bench/v100/s18check.py Logs/benchmarks/s110-f-fp16-8192-off.json Logs/benchmarks/s110-f-fp16-8192-on.json
  python3 bench/v100/s18check.py Logs/benchmarks/s110-f-int8-32768-off.json Logs/benchmarks/s110-f-int8-32768-on.json
  python3 - <<'EOF'
import json
for tag in ("s110-f-fp16-8192", "s110-f-int8-32768"):
    off = json.load(open(f"/home/noorazman/dsh/strata/Strata/Logs/benchmarks/{tag}-off.json"))
    on = json.load(open(f"/home/noorazman/dsh/strata/Strata/Logs/benchmarks/{tag}-on.json"))
    print(f"{tag}: OFF {off.get('decode_tps')} tok/s vs ON {on.get('decode_tps')} tok/s "
          f"(delta {100.0 * (on.get('decode_tps', 0) - off.get('decode_tps', 0)) / off.get('decode_tps', 1):+.1f} %)")
EOF
  echo "[$(date +%H:%M:%S)] PHASE F E2E DONE"
}

case "${1:-}" in
  ttft) phase_ttft ;;
  ctx) [ -n "${2:-}" ] || die "usage: s110.sh ctx <max-context>"; phase_ctx "$2" ;;
  ctxgate) [ -n "${2:-}" ] || die "usage: s110.sh ctxgate <max-context>"; phase_ctxgate "$2" ;;
  fence) phase_fence ;;
  *) sed -n '2,20p' "$0"; exit 1 ;;
esac
