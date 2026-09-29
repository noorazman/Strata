#!/usr/bin/env python3
"""Strata V100 Stage 1.1 benchmark orchestrator.

Runs one `strata generate` invocation on a chosen GPU while sampling:
  - nvidia-smi (VRAM, GPU/mem utilization, power, SM clock) at 10 Hz
  - /proc/stat per-core CPU utilisation at 10 Hz
  - /proc/<pid> thread count and per-thread CPU time at 1 Hz
  - /proc/meminfo (RAM) at start and end

The engine itself is launched with the canonical Stage 1 settings
(ulimit -l unlimited via sudo, LD_LIBRARY_PATH, pack/native/ple/profile/
mtp flags) plus the caller-supplied overrides.  Raw engine output, the
sampler CSVs and a parsed JSON summary are written under Logs/.

Usage (from the repo root):
  python3 bench/v100/bench.py <label> --gpu 0 --workers 28 --max-new 256
  python3 bench/v100/bench.py <label> --gpu 0 --workers 28 --max-new 16 \
      --tokens-file bench/v100/prefill_prompt.txt.tokens
  python3 bench/v100/bench.py <label> --gpu 0 --workers 28 --max-new 32 --stats
  python3 bench/v100/bench.py <label> --gpu 0 --workers 28 --strata-flags "--gpu-only-full"

Exit code: 0 when the engine ran to completion (its own exit code is
recorded in the JSON), 1 on harness failure.
"""

import argparse
import csv
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time

REPO = "/home/noorazman/dsh/strata/Strata"
SHARD1 = ("/mnt/ssd/llm_models/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF/"
          "Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf")
SUDO_PW = "mustoe8"
NCPU = os.cpu_count()
GPU_QUERY = ("memory.used,utilization.gpu,utilization.memory,power.draw,clocks.sm")


# ---------------------------------------------------------------- samplers

class GpuSampler(threading.Thread):
    """Polls nvidia-smi for one GPU at ~10 Hz into Logs/gpu/<label>.csv."""

    def __init__(self, label, gpu, stop):
        super().__init__(daemon=True)
        self.label, self.gpu, self.stop = label, gpu, stop
        self.peak_mem = 0
        self.peak_util = 0

    def run(self):
        path = f"{REPO}/Logs/gpu/{self.label}.csv"
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["t_s", "mem_mib", "gpu_util", "mem_util", "power_w", "sm_mhz"])
            q = ["nvidia-smi", "-i", str(self.gpu), "--query-gpu=" + GPU_QUERY,
                 "--format=csv,noheader,nounits"]
            while not self.stop.is_set():
                t0 = time.time()
                try:
                    r = subprocess.run(q, capture_output=True, text=True, timeout=2)
                    parts = [p.strip() for p in r.stdout.split(",")]
                    if len(parts) == 5:
                        mem, gu, mu, pw, mhz = (int(float(x)) for x in parts)
                        self.peak_mem = max(self.peak_mem, mem)
                        self.peak_util = max(self.peak_util, gu)
                        w.writerow([f"{t0:.3f}", mem, gu, mu, pw, mhz])
                except Exception:
                    pass
                time.sleep(max(0.0, 0.1 - (time.time() - t0)))


def _proc_stat():
    cores = {}
    with open("/proc/stat") as f:
        for line in f:
            if not line.startswith("cpu"):
                continue
            name, *rest = line.split()
            vals = list(map(int, rest[:8]))  # user nice system idle iowait irq softirq steal
            idle = vals[3] + vals[4]
            tot = sum(vals)
            cores[name] = (tot, idle)
    return cores


class CpuSampler(threading.Thread):
    """Per-core utilisation from /proc/stat at ~10 Hz into Logs/cpu/<label>.csv,
    plus 1 Hz process/thread detail into Logs/cpu/<label>.threads.csv."""

    def __init__(self, label, pid_getter, stop):
        super().__init__(daemon=True)
        self.label, self.pid_getter, self.stop = label, pid_getter, stop

    def run(self):
        path = f"{REPO}/Logs/cpu/{self.label}.csv"
        tpath = f"{REPO}/Logs/cpu/{self.label}.threads.csv"
        prev = _proc_stat()
        pt0 = time.time()
        prev_thread = None
        with open(path, "w", newline="") as f, open(tpath, "w", newline="") as tf:
            w = csv.writer(f)
            tw = csv.writer(tf)
            w.writerow(["t_s"] + [f"cpu{i}" for i in range(NCPU)] + ["total_pct"])
            tw.writerow(["t_s", "nthreads", "cpu_time_s", "cores_used", "max_thread_pct",
                         "thread_cores"])
            last_td = 0.0
            while not self.stop.is_set():
                t1 = time.time()
                cur = _proc_stat()
                pct = []
                busy_sum = 0.0
                wall = t1 - pt0
                for i in range(NCPU):
                    name = f"cpu{i}"
                    if name in prev and name in cur:
                        d_tot = cur[name][0] - prev[name][0]
                        d_idle = cur[name][1] - prev[name][1]
                        p = 100.0 * (d_tot - d_idle) / d_tot if d_tot > 0 else 0.0
                        busy_sum += p
                    else:
                        p = 0.0
                    pct.append(round(p, 1))
                prev, pt0 = cur, t1
                w.writerow([f"{t1:.3f}"] + pct + [round(busy_sum / NCPU, 1)])
                # 1 Hz thread detail
                pid = self.pid_getter()
                if pid is not None and (t1 - last_td) >= 1.0:
                    self._thread_detail(tw, t1, pid)
                    last_td = t1
                time.sleep(max(0.0, 0.1 - (time.time() - t1)))

    def _thread_detail(self, w, t1, pid):
        try:
            tids = [int(x) for x in os.listdir(f"/proc/{pid}/task")]
        except OSError:
            return
        cpu_s = 0.0
        cores = []
        best = (0.0, None)
        prev = getattr(self, "_prev_thread_time", None)
        now = {}
        for tid in tids:
            try:
                with open(f"/proc/{pid}/task/{tid}/stat") as tf2:
                    parts = tf2.read().rsplit(")", 1)[1].split()
                utime, stime = int(parts[11]), int(parts[12])
                proc = parts[36]
                now[tid] = utime + stime
                cpu_s += utime + stime
                cores.append(proc)
            except (OSError, IndexError):
                pass
        hz = os.sysconf("SC_CLK_TCK")
        wall = 1.0
        if prev:
            d = sum(v - prev.get(k, v) for k, v in now.items()) / hz
            best_pct = max((v - prev.get(k, 0)) / hz / wall * 100.0 for k, v in now.items()) \
                if now else 0.0
        else:
            best_pct = 0.0
        self._prev_thread_time = now
        from collections import Counter
        cnt = Counter(cores)
        w.writerow([f"{t1:.3f}", len(tids), f"{cpu_s / hz:.1f}",
                    f"{min(100.0, cpu_s / hz / wall * 100.0):.0f}", f"{best_pct:.0f}",
                    " ".join(f"{c}:{n}" for c, n in sorted(cnt.items()))])


# ---------------------------------------------------------------- engine

def gpu_free_mib(gpu):
    r = subprocess.run(["nvidia-smi", "-i", str(gpu),
                        "--query-gpu=memory.used,memory.total",
                        "--format=csv,noheader,nounits"],
                       capture_output=True, text=True)
    used, total = (int(x) for x in r.stdout.split(","))
    return total - used


def wait_idle(gpu, want_mib, timeout=90):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if gpu_free_mib(gpu) >= want_mib:
            return True
        time.sleep(5)
    return gpu_free_mib(gpu) >= want_mib


def engine_env(gpu, args_tail, env_extra=()):
    """The canonical Stage 1 launch, with the caller's tail appended.
    `env_extra` is a tuple of 'K=V' strings exported for the engine (Stage 1.5 A/B knobs)."""
    envs = " ".join(env_extra)
    # The env prefix goes BEFORE `timeout`: an env assignment is a shell feature, and `timeout 1500 VAR=x cmd`
    # would execvp the literal string "VAR=x" (rc 127). `VAR=x timeout ... cmd` exports it into the timeout
    # process, which the engine inherits.
    inner = (
        "ulimit -l unlimited; cd {repo} && "
        "CUDA_VISIBLE_DEVICES={gpu} LD_LIBRARY_PATH=/usr/local/cuda/lib64 "
        "{envs}timeout 1500 ./build-sm70/strata --pack packs/swift-iq3_xxs "
        "--native {shard1} --ple-gguf {shard1} "
        "--expert-profile data/expert-profile.bin {tail}"
    ).format(repo=REPO, gpu=gpu, shard1=SHARD1, tail=args_tail,
             envs=(envs + " ") if envs else "")
    return inner


def parse_engine(text):
    """Pull the metrics we care about out of the engine's stdout+stderr.

    Formats verified against the actual engine output (see
    Logs/benchmarks/phase0-baseline-32tok.log)."""
    out = {}

    def rx(pattern, key, cast=float, groups=1, flags=0):
        m = re.search(pattern, text, flags)
        if m:
            out[key] = cast(m.group(1)) if cast is not None else m.group(1)

    m = re.search(r"^\s*decode\s+(\d+) tokens in ([\d.]+) ms\s*->\s*([\d.]+) tok/s", text, re.M)
    if m:
        out.update(decode_tokens=int(m.group(1)), decode_ms=float(m.group(2)),
                   decode_tps=float(m.group(3)))
    m = re.search(r"^\s*prefill\s+(\d+) tokens in ([\d.]+) ms\s*->\s*([\d.]+) tok/s"
                  r"\s*\(time to first token ([\d.]+) ms\)", text, re.M)
    if m:
        out.update(prefill_tokens=int(m.group(1)), prefill_ms=float(m.group(2)),
                   prefill_tps=float(m.group(3)), ttft_ms=float(m.group(4)))
    m = re.search(r"^\s*per token\s+([\d.]+) ms/token", text, re.M)
    if m:
        out["wall_ms_tok"] = float(m.group(1))
    m = re.search(r"output\s*:\s*([0-9 ]+)", text)
    if m:
        out["output_tokens"] = [int(x) for x in m.group(1).split()]

    rx(r"^\s*expert cache auto: ([\d.]+) GiB free, (\d+) MiB reserved -> (\d+) slots",
       "cache_auto_free_gib", float, flags=re.M)
    m = re.search(r"^\s*expert cache auto: ([\d.]+) GiB free, (\d+) MiB reserved -> (\d+) slots",
                  text, re.M)
    if m:
        out.update(cache_auto_free_gib=float(m.group(1)), cache_reserve_mib=int(m.group(2)),
                   cache_slots=int(m.group(3)))
    m = re.search(r"^\s*expert cache (\d+) slots, ([\d.]+) GiB of VRAM", text, re.M)
    if m:
        out.update(cache_slots=int(m.group(1)), cache_vram_gib=float(m.group(2)))
    rx(r"arena[^;]*;? ([\d.]+) GiB loaded at ([\d.]+) GiB/s", "arena_gib", float, flags=re.M)
    rx(r"(\d+) expert-pool workers", "workers", int)
    m = re.search(r"strata mtp: draft layer loaded, (\d+) MiB of VRAM", text)
    if m:
        out["mtp_mib"] = float(m.group(1))
    m = re.search(r"experts streamed (\d+) \((\d+) by DMA, host ([\d.]+) ms\), resident (\d+);"
                  r" PLE ([\d.]+) ms", text)
    if m:
        out.update(streamed_experts=int(m.group(1)), dma_experts=int(m.group(2)),
                   stream_host_ms=float(m.group(3)), resident_experts=int(m.group(4)),
                   prefill_ple_ms=float(m.group(5)))
    m = re.search(r"ple io: (\d+) rows, ([\d.]+)% row-cache hits, (\d+) SSD reads"
                  r" \(([\d.]+) MB\), read p50 (\d+) us p99 (\d+) us, blocked ([\d.]+) ms total"
                  r" \(submit ([\d.]+) ms\)", text)
    if m:
        out.update(ple_rows=int(m.group(1)), ple_row_cache_pct=float(m.group(2)),
                   ple_ssd_reads=int(m.group(3)), ple_read_mb=float(m.group(4)),
                   ple_p50_us=int(m.group(5)), ple_p99_us=int(m.group(6)),
                   ple_blocked_ms=float(m.group(7)), ple_submit_ms=float(m.group(8)))
    # Stage 1.2A: RAM-resident PLE (--ple-io ram). No SSD reads; the stats line reports the preload.
    m = re.search(r"ple ram: (\d+) rows resident \(([\d.]+) GiB\), preloaded in ([\d.]+) s \(([\d.]+) GiB/s\);"
                  r" served (\d+) rows \(([\d.]+) MB\) from RAM", text)
    if m:
        out.update(ple_ram_rows=int(m.group(1)), ple_ram_gib=float(m.group(2)),
                   ple_ram_preload_s=float(m.group(3)), ple_ram_preload_gibs=float(m.group(4)),
                   ple_ram_served_rows=int(m.group(5)), ple_ram_served_mb=float(m.group(6)))
    m = re.search(r"speculation\s+(\d+) rounds of (\d+), drafts accepted (\d+) of (\d+)"
                  r" \(([\d.]+)\), ([\d.]+) tokens per round", text)
    if m:
        out.update(spec_rounds=int(m.group(1)), spec_window=int(m.group(2)),
                   spec_accepted=int(m.group(3)), spec_drafts=int(m.group(4)),
                   spec_accept=float(m.group(5)), spec_tpr=float(m.group(6)))
    m = re.search(r"^mtp\s+([\d.]+) ms/round drafting \((\d+) rounds\), MTP prompt ([\d.]+) ms,"
                  r" (\d+) MiB", text, re.M)
    if m:
        out.update(mtp_draft_ms_round=float(m.group(1)), mtp_rounds=int(m.group(2)),
                   mtp_prompt_ms=float(m.group(3)), mtp_vram_mib=int(m.group(4)))

    # --stats block
    m = re.search(r"token host phases.*?PLE ([\d.]+)\s+embed ([\d.]+)\s+LAYERS ([\d.]+)\s+"
                  r"head ([\d.]+)\s+readback ([\d.]+)\s+sample ([\d.]+)\s+\(sum ([\d.]+) of ([\d.]+) ms\)",
                  text)
    if m:
        out["host_phases_ms_tok"] = {k: float(v) for k, v in zip(
            ["ple", "embed", "layers", "head", "readback", "sample", "sum", "decode_wall"],
            m.groups())}
    m = re.search(r"^\s*the CPU expert pool\s+([\d.]+) ms/token over (\d+) layers"
                  r" \((\d+) positions, (\d+) dispatches\)", text, re.M)
    if m:
        out.update(pool_ms_tok=float(m.group(1)), pool_positions=int(m.group(3)))
    m = re.search(r"pool phases\s+wait-park ([\d.]+)\s+drain ([\d.]+)\s+re-park ([\d.]+)", text)
    if m:
        out["pool_phases_ms_tok"] = {"wait_park": float(m.group(1)),
                                     "drain": float(m.group(2)),
                                     "re_park": float(m.group(3))}
    m = re.search(r"^\s*expert blobs\s+(\d+) blobs read", text, re.M)
    if m:
        out["expert_blobs_read"] = int(m.group(1))
    m = re.search(r"R4 overlap\s+(\d+) of (\d+) layers", text)
    if m:
        out.update(r4_overlap_done=int(m.group(1)), r4_overlap_total=int(m.group(2)))
    m = re.search(r"R4 expert-cache hits\s+(\d+) of (\d+) = ([\d.]+)\s+"
                  r"\((\d+) admitted, (\d+) refused, cache ([\d.]+)% full\)", text)
    if m:
        out.update(cache_hits=int(m.group(1)), cache_lookups=int(m.group(2)),
                   cache_hit_rate=float(m.group(3)), cache_admitted=int(m.group(4)),
                   cache_refused=int(m.group(5)), cache_full_pct=float(m.group(6)))
    m = re.search(r"host after ring\s+([\d.]+) ms/token over (\d+) positions"
                  r" \(([\d.]+) ms/layer", text)
    if m:
        out.update(host_after_ring_ms_tok=float(m.group(1)), host_after_ring_layers=float(m.group(3)))
    m = re.search(r"ring latency\s+([\d.]+) ms of a ([\d.]+) ms layer", text)
    if m:
        out.update(ring_latency_ms=float(m.group(1)), layer_ms=float(m.group(2)))
    # --gpu-stages / --gpu-only-full tables
    m = re.search(r"per-stage GPU time on the CAPTURED graph.*?sum of the three\s+([\d.]+) ms",
                  text, re.S)
    if m:
        out["gpu_stages_total_ms_tok"] = float(m.group(1))
    m = re.search(r"mixer \(gr_read\+attn\+gr_write\)\s+([\d.]+) ms/token\s+[\d.]+\s+[\d.]+%", text)
    if m:
        out["gpu_mixer_ms_tok"] = float(m.group(1))
    m = re.search(r"ffn front \+ router\s+([\d.]+) ms/token", text)
    if m:
        out["gpu_ffn_ms_tok"] = float(m.group(1))
    m = re.search(r"post \(moe_finish\+gr_write\)\s+([\d.]+) ms/token", text)
    if m:
        out["gpu_post_ms_tok"] = float(m.group(1))
    m = re.search(r"^\s*gpu-only-full\s+([\d.]+) ms", text, re.M)
    if m:
        out["gpu_only_full_ms"] = float(m.group(1))
    m = re.search(r"GPU floor[^.]*?([\d.]+) ms", text)
    if m:
        out.setdefault("gpu_only_full_ms", float(m.group(1)))
    if re.search(r"NOT CORRECT", text):
        out["hit_path_warning"] = True
    return out


def cpu_summary(path):
    """Aggregate the per-core CSV: mean/peak total, mean/peak per-core, hot cores."""
    try:
        rows = list(csv.DictReader(open(path)))
    except OSError:
        return {}
    if not rows:
        return {}
    total = [float(r["total_pct"]) for r in rows]
    per_core = [[] for _ in range(NCPU)]
    for r in rows:
        for i in range(NCPU):
            per_core[i].append(float(r[f"cpu{i}"]))
    mean = lambda xs: sum(xs) / len(xs) if xs else 0.0
    hot = []
    for i, xs in enumerate(per_core):
        m = mean(xs)
        if m >= 50:
            hot.append(f"{i}:{m:.0f}%")
    return {"cpu_total_mean_pct": round(mean(total), 1),
            "cpu_total_peak_pct": round(max(total), 1),
            "cpu_core_mean_pct": round(mean([mean(xs) for xs in per_core]), 1),
            "cpu_core_peak_pct": round(max(max(xs) for xs in per_core), 1),
            "cores_mean_ge50pct": hot}


def meminfo():
    d = {}
    for line in open("/proc/meminfo"):
        k, v = line.split(":", 1)
        d[k] = int(v.split()[0])  # kB
    return {"mem_total_gib": d.get("MemTotal", 0) / 1048576,
            "mem_available_gib": d.get("MemAvailable", 0) / 1048576,
            "cached_gib": d.get("Cached", 0) + d.get("SReclaimable", 0) / 1048576}


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("label")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--workers", type=int, default=28)
    ap.add_argument("--max-new", type=int, default=32)
    ap.add_argument("--max-context", type=int, default=8192)
    ap.add_argument("--kv", default="fp16", choices=["fp16", "int8"])
    ap.add_argument("--tokens-file", default=None)
    ap.add_argument("--tokens", default=None)
    ap.add_argument("--stats", action="store_true")
    ap.add_argument("--no-samplers", action="store_true",
                    help="skip the CPU/GPU samplers (measurement-only runs)")
    ap.add_argument("--timeout", type=int, default=1500)
    ap.add_argument("--strata-flags", default="",
                    help="verbatim extra strata flags, e.g. --strata-flags \"--gpu-only-full\"")
    ap.add_argument("--env", action="append", default=[],
                    help="export K=V for the engine; repeatable (Stage 1.5 A/B knobs, "
                         "e.g. --env STRATA_POOL_PARK=64)")
    a = ap.parse_args()

    if not (a.tokens or a.tokens_file):
        a.tokens = ("9419,11,821,803,369,7967,13,353,1044,264,11952,5617,303,220,17,"
                    "15,17,21,13,10875,353,668,3184,488,883,821,1118,1834,13")
    # Long prompts: pass --tokens-file straight to the engine (native file read) instead of a
    # --tokens command argument (MAX_ARG_STRLEN = 128 KB caps a single argv string).
    tok_arg = f"--tokens-file {a.tokens_file}" if a.tokens_file else f"--tokens {a.tokens}"
    if a.tokens_file:
        a.tokens = open(a.tokens_file).read().strip()

    tail = (f"--expert-cache auto --prefill 2048 --spec 4 --spec-min-p 0.5 "
            f"--pool-workers {a.workers} --mtp mtp/rt --max-context {a.max_context} "
            f"--kv {a.kv} --max-new {a.max_new} {tok_arg}")
    if a.stats:
        tail += " --stats"
    if a.strata_flags:
        tail += " " + a.strata_flags

    for sub in ("gpu", "cpu", "benchmarks"):
        os.makedirs(f"{REPO}/Logs/{sub}", exist_ok=True)

    free = gpu_free_mib(a.gpu)
    print(f"[bench] {a.label}: gpu{a.gpu} free {free} MiB; "
          f"{'waiting for idle' if free < 2000 else 'ok'}")
    if free < 2000 and not wait_idle(a.gpu, 2000):
        print(f"[bench] {a.label}: gpu{a.gpu} still busy, aborting")
        return 1
    mem_before = meminfo()
    time.sleep(2)  # let the page cache settle its numbers

    stop = threading.Event()
    gpu_s, cpu_s = None, None
    if not a.no_samplers:
        gpu_s = GpuSampler(a.label, a.gpu, stop)
        gpu_s.start()

    pid_box = {"pid": None}

    def pid_getter():
        return pid_box["pid"]

    if not a.no_samplers:
        cpu_s = CpuSampler(a.label, pid_getter, stop)
        cpu_s.start()

    inner = engine_env(a.gpu, tail, env_extra=tuple(a.env or ()))
    t0 = time.time()
    proc = subprocess.Popen(
        ["sudo", "-S", "-p", "", "bash", "-c", inner],
        stdin=subprocess.PIPE, stdout=open(f"{REPO}/Logs/benchmarks/{a.label}.log", "wb"),
        stderr=subprocess.STDOUT)
    proc.stdin.write(SUDO_PW.encode() + b"\n")
    proc.stdin.close()

    # grab the engine pid for the CPU sampler (strata under sudo)
    while proc.poll() is None:
        time.sleep(0.5)
        r = subprocess.run(["pgrep", "-x", "strata"], capture_output=True, text=True)
        pids = [int(x) for x in r.stdout.split()]
        if pids:
            pid_box["pid"] = max(pids)
            break
    rc = proc.wait()
    wall = time.time() - t0
    stop.set()
    time.sleep(0.4)
    if gpu_s:
        gpu_s.join(timeout=3)
    if cpu_s:
        cpu_s.join(timeout=3)

    text = open(f"{REPO}/Logs/benchmarks/{a.label}.log", "r", errors="replace").read()
    summary = parse_engine(text)
    summary.update({
        "label": a.label,
        "gpu": a.gpu,
        "workers": a.workers,
        "max_new": a.max_new,
        "max_context": a.max_context,
        "kv": a.kv,
        "n_prompt": len(a.tokens.split(",")),
        "wall_s": round(wall, 1),
        "engine_rc": rc,
        "peak_vram_mib": gpu_s.peak_mem if gpu_s else None,
        "peak_gpu_util": gpu_s.peak_util if gpu_s else None,
        "mem_before": mem_before,
        "mem_after": meminfo(),
        "command_tail": tail,
    })
    summary["cpu"] = cpu_summary(f"{REPO}/Logs/cpu/{a.label}.csv")

    out_path = f"{REPO}/Logs/benchmarks/{a.label}.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=1)

    d = summary
    line = (f"[bench] {a.label}: rc={rc} wall={d['wall_s']}s "
            f"decode={d.get('decode_tps', '-')} tok/s "
            f"({d.get('decode_tokens', '?')} tok / {d.get('decode_ms', '?')} ms) "
            f"prefill={d.get('prefill_tps', '-')} tok/s "
            f"TTFT={d.get('ttft_ms', '-')} ms "
            f"peakVRAM={d.get('peak_vram_mib')} MiB "
            f"cpuMean={d['cpu'].get('cpu_total_mean_pct', '-')}% "
            f"peakGPU={d.get('peak_gpu_util')}%")
    print(line)
    if d.get("cache_hit_rate") is not None:
        print(f"[bench]   cache hits {d['cache_hits']}/{d['cache_lookups']} "
              f"= {100*d['cache_hit_rate']:.1f}% (admitted {d['cache_admitted']}, "
              f"refused {d['cache_refused']}, {d['cache_full_pct']}% full)")
    if d.get("spec_tpr") is not None:
        print(f"[bench]   spec: {d['spec_rounds']} rounds x {d['spec_window']}, "
              f"accepted {d['spec_accepted']}/{d['spec_drafts']} ({d['spec_accept']}), "
              f"{d['spec_tpr']} tok/round; mtp draft {d.get('mtp_draft_ms_round', '?')} ms/round")
    if d.get("ple_blocked_ms") is not None:
        print(f"[bench]   PLE: {d['ple_rows']} rows, {d['ple_ssd_reads']} SSD reads "
              f"({d['ple_read_mb']} MB), p50 {d['ple_p50_us']}us, blocked {d['ple_blocked_ms']} ms")
    if d.get("ple_ram_gib") is not None:
        print(f"[bench]   PLE RAM: {d['ple_ram_gib']} GiB resident, preloaded in {d['ple_ram_preload_s']} s "
              f"({d['ple_ram_preload_gibs']} GiB/s); served {d['ple_ram_served_rows']} rows "
              f"({d['ple_ram_served_mb']} MB) from RAM")
    if d.get("host_phases_ms_tok"):
        hp = d["host_phases_ms_tok"]
        print(f"[bench]   host phases ms/tok: PLE {hp['ple']:.2f} embed {hp['embed']:.2f} "
              f"layers {hp['layers']:.2f} head {hp['head']:.2f} "
              f"readback {hp['readback']:.2f} sample {hp['sample']:.2f} "
              f"= {hp['sum']:.2f} of {hp['decode_wall']:.2f}")
    if d.get("pool_ms_tok"):
        pp = d.get("pool_phases_ms_tok", {})
        print(f"[bench]   pool {d['pool_ms_tok']:.2f} ms/tok "
              f"(drain {pp.get('drain', '?')}, re-park {pp.get('re_park', '?')}, "
              f"wait {pp.get('wait_park', '?')})")
    if d.get("output_tokens"):
        print(f"[bench]   output: {' '.join(map(str, d['output_tokens'][:16]))} ... "
              f"({len(d['output_tokens'])} tokens)")
    return 0 if rc == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
