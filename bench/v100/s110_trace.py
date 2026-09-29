#!/usr/bin/env python3
"""Stage 1.10 — flag-A phase attribution: prefill vs first-token (round 0) vs normal decode.

Uses the same kernel-level A/B/C classifier as s19_trace.py (anchored on the plan copy that
follows every wait-A) and buckets every wait_flag_ge instance by the request phase it belongs to:

    prefill   waits with start < the first doorbell_publish_kernel  (the prefill path has NO
              wait_flag_ge by construction; this verifies it)
    round 0   waits between the first doorbell and the first sampler_greedy launch — the FIRST
              verify window, i.e. the first-token generation (the window that ends TTFT)
    decode    waits after the first sampler (rounds 1..N-1, normal decode)

and reports, per phase: wait count, total/mean/max, the A/B/C split, the >1 ms cohort, and the
per-round dispatch-0 (round-head) A lag — the Stage 1.9 boundary-visibility signature — so the
question "does the round-head flag-A lag hit prefill / the first token / normal decode, and does
STRATA_WAIT_FENCE=1 move it" is answered from the same capture.

Usage:
    s110_trace.py <capture.sqlite> [decode_tokens] [--json out.json]
"""
import bisect
import json
import sqlite3
import sys
from collections import defaultdict

sys.path.insert(0, "/home/noorazman/dsh/strata/Strata/bench/v100")
from s19_trace import base, classify, load, pct  # noqa: E402


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    path = sys.argv[1]
    ntok = int(sys.argv[2]) if len(sys.argv) > 2 else 64
    js = sys.argv[sys.argv.index("--json") + 1] if "--json" in sys.argv else None

    k, m = load(path)
    for e in k:
        e["b"] = base(e["name"])

    # the verify-window stream carries the doorbells (one per dispatch per round).  In the 256-tok
    # generate captures it is also the busiest stream; in serve captures the PREFILL path owns the
    # busiest stream, so the doorbell count is the reliable identifier.
    scnt = defaultdict(int)
    dcnt = defaultdict(int)
    for e in k:
        scnt[e["streamId"]] += 1
        if base(e["name"]) == "doorbell_publish_kernel":
            dcnt[e["streamId"]] += 1
    if dcnt:
        main_stream = max(dcnt, key=dcnt.get)
    else:
        main_stream = max(scnt, key=scnt.get)
    seq = sorted((e for e in k if e["streamId"] == main_stream), key=lambda e: e["start"])

    doorbells = [e["start"] for e in seq if e["b"] == "doorbell_publish_kernel"]
    samplers = [e["start"] for e in seq if e["b"] in ("sample_tokens", "sampler_kernel", "sampler_greedy_kernel")]
    if not doorbells or not samplers:
        sys.exit("no doorbell/sampler found on the main stream - wrong capture?")
    r0_start, r0_end = doorbells[0], samplers[0]
    k0 = seq[0]["start"]
    k1 = seq[-1]["end"]

    waits, nanchors, anomalies = classify(seq)
    total_wait = sum(w["end"] - w["start"] for w in waits)
    main_busy = sum(e["end"] - e["start"] for e in seq)

    def phase_of(w):
        if w["start"] < r0_start:
            return "prefill"
        if w["start"] < r0_end:
            return "round0"
        return "decode"

    for w in waits:
        w["phase"] = phase_of(w)
        if w["phase"] == "decode":
            w["round"] = bisect.bisect_left(samplers, w["start"])
        else:
            w["round"] = 0 if w["phase"] == "round0" else -1

    # dispatch position within the round (waits arrive in A/B/C triples per dispatch)
    per_round = defaultdict(list)
    for w in waits:
        per_round[(w["phase"], w["round"])].append(w)
    for _, ws in per_round.items():
        for i, w in enumerate(ws):
            w["d"] = i // 3

    hist_bins = [2, 10, 50, 100, 500, 1000, 5000]

    def line(tag, ws):
        d = [w["end"] - w["start"] for w in ws]
        if not d:
            print(f"{tag:10s} n=0")
            return None
        t = sum(d)
        hs, lo = [], 0
        for hi in hist_bins:
            hs.append(len([x for x in d if lo <= x < hi]))
            lo = hi
        hs.append(len([x for x in d if x >= lo]))
        print(f"{tag:10s} n={len(d):6d} {t/1e6:8.1f} ms  mean {t/len(d)/1e3:8.1f} us  "
              f"p50 {pct(d,50)/1e3:7.1f}  p99 {pct(d,99)/1e3:7.1f}  max {max(d)/1e3:8.1f} us  hist {hs}")
        return {"n": len(d), "ms": round(t / 1e6, 1), "mean_us": round(t / len(d) / 1e3, 2),
                "max_us": round(max(d) / 1e3, 1), "hist": hs}

    print(f"== s110 flag-A phase attribution ({path.split('/')[-1]}) ==")
    print(f"main stream {main_stream} ({len(seq)} kernels, {main_busy/1e6:.0f} ms busy); "
          f"trace span {(k1-k0)/1e6:.0f} ms")
    print(f"prefill phase: [{k0/1e6:.0f}, {r0_start/1e6:.0f}) ms; round 0 (first token): "
          f"[{r0_start/1e6:.0f}, {r0_end/1e6:.0f}) ms; decode: after {r0_end/1e6:.0f} ms")
    print(f"waits: {len(waits)} total, anchors {nanchors}, anomalies {dict(anomalies) or 'none'}")
    print(f"total wait {total_wait/1e6:.1f} ms = {total_wait/1e3/max(ntok,1):.0f} us/{ntok} tok")

    out = {"path": path, "decode_tokens": ntok, "total_wait_ms": round(total_wait / 1e6, 1),
           "anchors": nanchors, "anomalies": dict(anomalies), "phases": {}}
    for ph in ("prefill", "round0", "decode"):
        ws = [w for w in waits if w["phase"] == ph]
        print(f"\n--- phase {ph}: {len(ws)} waits")
        line(f"{ph} all", ws)
        per_flag = {}
        for f in "ABC":
            r = line(f"{ph} {f}", [w for w in ws if w["flag"] == f])
            if r:
                per_flag[f] = r
        big = [w for w in ws if w["end"] - w["start"] > 1_000_000]
        bt = sum(w["end"] - w["start"] for w in big)
        print(f"{ph:10s} >1ms cohort: {len(big)} instances, {bt/1e6:.1f} ms, "
              f"max {max((w['end']-w['start'] for w in big), default=0)/1e3:.1f} ms")
        out["phases"][ph] = {"waits": len(ws), "total_ms": round(sum(w["end"] - w["start"] for w in ws) / 1e6, 1),
                             "flags": per_flag, "gt1ms": {"n": len(big), "ms": round(bt / 1e6, 1),
                                                           "max_ms": round(max((w["end"] - w["start"] for w in big), default=0) / 1e3, 1)}}

    # per-round round-head (dispatch 0) A lag: the Stage 1.9 boundary signature, now per round
    head_a = []
    for rr in sorted(set(w["round"] for w in waits if w["phase"] == "decode") | {0}):
        ws = [w for w in waits if w["round"] == rr and w.get("d") == 0 and w["flag"] == "A"]
        if ws:
            head_a.append((rr, ws[0]["end"] - ws[0]["start"]))
    head_a.sort()
    if head_a:
        vals = [v for _, v in head_a]
        print(f"\nround-head A per round (round: ms): "
              + " ".join(f"{r}={v/1e3:.1f}" for r, v in head_a[:40])
              + (" ..." if len(head_a) > 40 else ""))
        over = [(r, v) for r, v in head_a if v > 1_000_000]
        print(f"round-head A > 1 ms: {len(over)}/{len(head_a)} rounds; "
              f"mean {sum(vals)/len(vals)/1e6:.2f} ms; max {max(vals)/1e6:.1f} ms; "
              f"round 0 = {next((v/1e6 for r, v in head_a if r == 0), float('nan')):.1f} ms  "
              f"(per-round values above are us)")
        out["round_head_A"] = {"n_rounds": len(head_a), "gt1ms_rounds": len(over),
                               "mean_ms": round(sum(vals) / len(vals) / 1e6, 2),
                               "max_ms": round(max(vals) / 1e6, 1),
                               "round0_ms": round(next((v / 1e6 for r, v in head_a if r == 0), float('nan')), 1)
                               if any(r == 0 for r, _ in head_a) else None,
                               "per_round_us": [[r, round(v / 1e3, 1)] for r, v in head_a]}
    else:
        print("\nno round-head A waits found")
    if js:
        json.dump(out, open(js, "w"), indent=1)
        print(f"json -> {js}")


if __name__ == "__main__":
    main()
