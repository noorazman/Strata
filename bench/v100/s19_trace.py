#!/usr/bin/env python3
"""Stage 1.9 — wait_flag_ge A/B/C attribution for an nsys sqlite capture
(cuda-graph-trace=node).  Profile-only: no engine changes.

The verify-window graph runs per-dispatch (per layer l, G=1) on ONE main stream:

    pre(l):  ... layer kernels ... moe_route, doorbell_publish
    post(l): wait_flag_ge (A)      <- host published the GPU plan (flagA, non-posted)
           copy_i32_from_mapped    (the plan, read over PCIe)
           native_expert_grouped   (VRAM-resident experts)
           wait_flag_ge (B)        <- staging DMA landed (flagB, copy-stream hostfunc)
           native_expert_grouped   (PCIe-staged experts)
           wait_flag_ge (C)        <- the CPU pool's rows are in (flagC, pool worker)
           copy_from_mapped        (the rows, read over PCIe)
           moe_hit_add, moe_combine, [gr_write on the last layer]

The three waits are POSITIONAL (A, B, C in that order, 3 per dispatch, 48 per
round).  This script classifies each wait_flag_ge instance by anchoring on the
(plan) copy_i32_from_mapped that immediately follows every wait-A, then filling
the B/C positions between anchors; it reports anchor coverage as a confidence
check.  Round boundaries are the sampler kernels (one sample_tokens launch per
round; the k-th sampler sits between round k's l47 wait-C and round k+1's l0
wait-A).

Outputs (also as JSON with --json):
  - per-flag (A/B/C): count, total ms, ms/tok, mean/p50/p90/p99/max, histogram
  - round-head (dispatch 0) vs mid-round breakdown
  - the >1 ms wait cohort (Stage 1.6 round-head visibility lag)
  - gap analysis:
      waitA: waitA.start - preceding doorbell_publish.end (doorbell visibility + plan)
      waitB: staging H2D memcpy completion vs waitB end (the DMA-tie check)
  - per-round: wait total, A/B/C split, staged-blob count

Usage: s19_trace.py <capture.sqlite> [decode_tokens] [--json out.json]
"""
import json
import re
import sqlite3
import sys
from collections import defaultdict


def load(path):
    db = sqlite3.connect(path)
    db.row_factory = sqlite3.Row
    cur = db.cursor()
    k = [dict(r) for r in cur.execute(
        "SELECT start, end, streamId, s.value AS name FROM CUPTI_ACTIVITY_KIND_KERNEL k "
        "LEFT JOIN StringIds s ON s.id = k.demangledName ORDER BY start")]
    m = [dict(r) for r in cur.execute(
        "SELECT start, end, streamId, bytes, copyKind, o.name AS kind FROM CUPTI_ACTIVITY_KIND_MEMCPY m "
        "LEFT JOIN ENUM_CUDA_MEMCPY_OPER o ON o.id = m.copyKind ORDER BY start")]
    return k, m


def base(name):
    if not name:
        return "<null>"
    name = re.sub(r"^void\s+", "", name)
    m = re.search(r"([a-zA-Z_][a-zA-Z0-9_]*)(<|\(|\s)", name)
    return m.group(1) if m else name


def find_decode_window(k):
    gaps = []
    prev_end = None
    for e in k:
        if prev_end is not None:
            g = e["start"] - prev_end
            if g >= 2_000_000:
                gaps.append((g, e["start"]))
        prev_end = max(prev_end or 0, e["end"])
    if not gaps:
        return k[0]["start"], k[-1]["end"]
    gaps.sort(reverse=True)
    return gaps[0][1], k[-1]["end"]


def pct(xs, p):
    if not xs:
        return 0.0
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(p / 100.0 * (len(xs) - 1))))]


def classify(seq):
    """Classify each wait_flag_ge as A/B/C.

    The post(l) pattern on the main stream is, from the source (verify.cpp):
      waitA, copy_i32_from_mapped, [the VRAM grouped pipeline: 0 or 4 kernels],
      waitB, [the PCIe grouped pipeline: 0 or 4 kernels], waitC,
      copy_from_mapped, add_hits, moe_combine...
    so the (plan) copy_i32_from_mapped that immediately follows a wait uniquely
    identifies every wait-A; every second wait after an anchor is B, the next C,
    and the cycle repeats.  The grouped pipeline is 0 kernels when the group is
    empty (native_expert_grouped returns early), so kernel counts in between
    are NOT a reliable signature - the waits themselves are.
    Returns (waits, n_anchors, anomalies).
    """
    n = len(seq)
    wait_pos = []
    for i in range(n):
        if seq[i]["b"] == "wait_flag_ge_kernel":
            wait_pos.append(i)
    nw = len(wait_pos)
    # anchor: a wait followed within 2 events by copy_i32_from_mapped_kernel
    anchors = []
    for a, i in enumerate(wait_pos):
        for j in range(i + 1, min(i + 3, n)):
            if seq[j]["b"] == "copy_i32_from_mapped_kernel":
                anchors.append(a)
                break
    anomalies = defaultdict(int)
    if not anchors:
        anomalies["no-anchors"] += 1
        anchors = list(range(0, nw, 3))
    else:
        if anchors[0] != 0:
            anomalies["leading-waits"] += anchors[0]
        for k in range(len(anchors) - 1):
            gap = anchors[k + 1] - anchors[k]
            if gap != 3:
                anomalies["anchor-gap-%d" % gap] += 1
        trail = nw - anchors[-1] - 1
        if trail:
            anomalies["trailing-waits"] += trail
    # sanity: after every C (anchor+2) the rows copy must follow within 3 events
    for a in anchors[:-1] if (nw - anchors[-1] - 1) else anchors:
        c = a + 2
        if c < nw:
            i = wait_pos[c]
            ok = any(seq[j]["b"] == "copy_from_mapped_kernel" for j in range(i + 1, min(i + 4, n)))
            if not ok:
                anomalies["C-without-rows-copy"] += 1
    phase = {}
    for k, a in enumerate(anchors):
        phase[a] = 0
        if k + 1 < len(anchors):
            for off in range(1, anchors[k + 1] - a):
                phase[a + off] = off % 3
        else:
            for off in range(1, nw - a):
                phase[a + off] = off % 3
    for off in range(anchors[0]):
        phase[off] = (off - anchors[0]) % 3
    flags = "ABC"
    waits = []
    for a, i in enumerate(wait_pos):
        e = seq[i]
        waits.append({"start": e["start"], "end": e["end"], "flag": flags[phase.get(a, a % 3)], "a": a})
    return waits, len(anchors), anomalies


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    path = sys.argv[1]
    ntok = int(sys.argv[2]) if len(sys.argv) > 2 else 256
    js = sys.argv[sys.argv.index("--json") + 1] if "--json" in sys.argv else None

    k, m = load(path)
    d0, d1 = find_decode_window(k)
    span = d1 - d0
    dec = [e for e in k if e["start"] >= d0]
    mem = [e for e in m if e["start"] >= d0]
    for e in dec:
        e["b"] = base(e["name"])

    scnt = defaultdict(int)
    for e in dec:
        scnt[e["streamId"]] += 1
    main_stream = max(scnt, key=scnt.get)
    seq = sorted((e for e in dec if e["streamId"] == main_stream), key=lambda e: e["start"])

    h2d = sorted((e for e in mem if e["kind"] and "HTOD" in e["kind"]
                  and e["streamId"] != main_stream and e["bytes"] >= 256 * 1024), key=lambda e: e["start"])
    samplers = [e["start"] for e in seq if e["b"] in ("sample_tokens", "sampler_kernel", "sampler_greedy_kernel")]
    doorbells = [e for e in seq if e["b"] == "doorbell_publish_kernel"]

    waits, nanchors, anomalies = classify(seq)

    # rounds: the k-th sampler ends round k; a wait is in round k iff k samplers
    # started before it.
    import bisect
    for w in waits:
        w["round"] = bisect.bisect_left(samplers, w["start"])
    per_round = defaultdict(list)
    for w in waits:
        per_round[w["round"]].append(w)
    for r, ws in per_round.items():
        for i, w in enumerate(ws):
            w["d"] = i // 3
            w["in3"] = i % 3
    nrounds = max(per_round, default=-1) + 1

    total_wait = sum(w["end"] - w["start"] for w in waits)
    main_busy = sum(e["end"] - e["start"] for e in seq)
    out = {"path": path, "tokens": ntok, "window_ms": round(span / 1e6, 1),
           "total_wait_ms": round(total_wait / 1e6, 1), "us_per_tok": round(total_wait / 1e3 / ntok, 0),
           "main_stream_kernel_ms": round(main_busy / 1e6, 1),
           "n_waits": len(waits), "anchors": nanchors, "anomalies": dict(anomalies),
           "rounds": nrounds, "rounds_by_sampler": len(samplers), "flags": {}, "per_round": {}}

    print(f"== s19 wait_flag_ge attribution ({path.split('/')[-1]}) ==")
    print(f"decode window {span/1e6:.0f} ms, {len(dec)} kernels, main stream {main_stream} ({len(seq)} ev), "
          f"H2D>=256KiB copy-stream: {len(h2d)}")
    print(f"wait instances {len(waits)}; anchors (wait followed by plan-copy) {nanchors}/3-expected "
          f"({100*nanchors/max(len(waits),1):.0f}%); anomalies: {dict(anomalies) or 'none'}")
    print(f"rounds: {nrounds} by wait grouping, {len(samplers)} sampler launches")
    if any(w.get("d", 0) >= 48 for w in waits):
        print("WARNING: a round has >=144 waits (>=48 dispatches) - check the round boundary")
    print(f"total wait {total_wait/1e6:.1f} ms = {total_wait/1e3/ntok:.0f} us/tok = "
          f"{100*total_wait/max(main_busy,1):.1f}% of main-stream kernel time")

    hist_bins = [2, 10, 50, 100, 500, 1000, 5000]
    print(f"\n{'flag':4s} {'n':>7s} {'ms':>8s} {'us/tok':>8s} {'mean':>8s} {'p50':>8s} {'p90':>8s} {'p99':>8s} {'max':>9s}  hist(<=2,2-10,10-50,50-100,100-500,500-1k,1-5k,>5k us)")
    for f in "ABC":
        ws = [w for w in waits if w["flag"] == f]
        d = [w["end"] - w["start"] for w in ws]
        t = sum(d)
        hs = []
        lo = 0
        for hi in hist_bins:
            hs.append(len([x for x in d if lo <= x < hi]))
            lo = hi
        hs.append(len([x for x in d if x >= lo]))
        if d:
            print(f"{f:4s} {len(d):>7d} {t/1e6:>8.1f} {t/1e3/ntok:>8.0f} {t/len(d)/1e3:>8.1f} "
                  f"{pct(d,50)/1e3:>8.1f} {pct(d,90)/1e3:>8.1f} {pct(d,99)/1e3:>8.1f} {max(d)/1e3:>9.1f}  {hs}")
        out["flags"][f] = {"n": len(d), "ms": round(t / 1e6, 1), "us_per_tok": round(t / 1e3 / ntok, 1),
                           "mean_us": round(t / len(d) / 1e3, 2) if d else 0,
                           "p50_us": round(pct(d, 50) / 1e3, 2) if d else 0,
                           "p99_us": round(pct(d, 99) / 1e3, 2) if d else 0,
                           "max_us": round(max(d) / 1e3, 1) if d else 0, "hist": hs}

    head = [w for w in waits if w.get("d", 0) == 0]
    mid = [w for w in waits if w.get("d", 0) > 0]
    print(f"\nround-head (dispatch 0) vs mid-round:")
    for f in "ABC":
        h = [w["end"] - w["start"] for w in head if w["flag"] == f]
        mi = [w["end"] - w["start"] for w in mid if w["flag"] == f]
        print(f"  {f}: head n={len(h)} {sum(h)/1e6:.1f} ms (mean {sum(h)/max(len(h),1)/1e3:.1f} us, max {max(h)/1e3 if h else 0:.1f} us)  |  "
              f"mid n={len(mi)} {sum(mi)/1e6:.1f} ms (mean {sum(mi)/max(len(mi),1)/1e3:.1f} us)")
    out["head_mid"] = {f: {"head_ms": round(sum(w["end"] - w["start"] for w in head if w["flag"] == f) / 1e6, 1),
                           "mid_ms": round(sum(w["end"] - w["start"] for w in mid if w["flag"] == f) / 1e6, 1)} for f in "ABC"}

    big = sorted((w for w in waits if w["end"] - w["start"] > 1_000_000), key=lambda w: -(w["end"] - w["start"]))
    bt = sum(w["end"] - w["start"] for w in big)
    bf = defaultdict(lambda: [0, 0.0])
    for w in big:
        bf[w["flag"]][0] += 1
        bf[w["flag"]][1] += w["end"] - w["start"]
    print(f"\nwaits > 1 ms: {len(big)} instances, {bt/1e6:.1f} ms ({100*bt/max(total_wait,1):.1f}% of all wait time), "
          f"max {max((w['end']-w['start'] for w in big), default=0)/1e3:.1f} ms")
    for f in "ABC":
        c, t = bf.get(f, [0, 0.0])
        print(f"  {f}: {c} instances, {t/1e6:.1f} ms")
    head_big = [w for w in big if w.get("d", 0) == 0]
    mid_big = [w for w in big if w.get("d", 0) > 0]
    print(f"  by position: round-head {len(head_big)} ({sum(w['end']-w['start'] for w in head_big)/1e6:.1f} ms), "
          f"mid-round {len(mid_big)} ({sum(w['end']-w['start'] for w in mid_big)/1e6:.1f} ms)")
    print(f"  top 15:")
    for w in big[:15]:
        print(f"    round {w['round']:>3d} l{w['d']:>2d} {w['flag']}: {(w['end']-w['start'])/1e3:8.1f} ms  (t={(w['start']-d0)/1e6:7.1f} ms into window)")
    out["big1ms"] = {"n": len(big), "ms": round(bt / 1e6, 1),
                     "by_flag": {f: [bf.get(f, [0, 0.0])[0], round(bf.get(f, [0, 0.0])[1] / 1e6, 1)] for f in "ABC"},
                     "head_n": len(head_big), "head_ms": round(sum(w["end"] - w["start"] for w in head_big) / 1e6, 1),
                     "top": [{"round": w["round"], "l": w["d"], "flag": w["flag"],
                              "ms": round((w["end"] - w["start"]) / 1e3, 1)} for w in big[:20]]}

    lay = defaultdict(float)
    for w in waits:
        lay[w.get("d", -1)] += w["end"] - w["start"]
    print(f"\nworst layers by total wait (ms over the window):")
    for l, t in sorted(lay.items(), key=lambda kv: -kv[1])[:12]:
        c = len([w for w in waits if w.get("d") == l])
        print(f"  l{l:>2d}: {t/1e6:7.1f} ms over {c} waits (mean {t/max(c,1)/1e3:.1f} us)")
    out["worst_layers"] = [{"l": l, "ms": round(t / 1e6, 1)} for l, t in sorted(lay.items(), key=lambda kv: -kv[1])[:12]]

    # waitA host-side gap: preceding doorbell end -> waitA start
    dbi = 0
    gapsA = []
    for w in waits:
        if w["flag"] != "A":
            continue
        while dbi + 1 < len(doorbells) and doorbells[dbi + 1]["start"] < w["start"]:
            dbi += 1
        if dbi < len(doorbells) and doorbells[dbi]["start"] < w["start"]:
            gapsA.append(w["start"] - doorbells[dbi]["end"])
    if gapsA:
        big_g = [g for g in gapsA if g > 1_000_000]
        print(f"\nwaitA start - preceding doorbell end (host: doorbell visibility + plan + A publish): "
              f"n={len(gapsA)} mean {sum(gapsA)/len(gapsA)/1e3:.1f} us p50 {pct(gapsA,50)/1e3:.1f} us "
              f"p90 {pct(gapsA,90)/1e3:.1f} us p99 {pct(gapsA,99)/1e3:.1f} us max {max(gapsA)/1e3:.1f} us")
        print(f"  >1 ms: {len(big_g)} ({sum(big_g)/1e6:.1f} ms)")
        out["gapA_host_us"] = {"n": len(gapsA), "mean": round(sum(gapsA) / len(gapsA) / 1e3, 1),
                               "p50": round(pct(gapsA, 50) / 1e3, 1), "p99": round(pct(gapsA, 99) / 1e3, 1),
                               "max_us": round(max(gapsA) / 1e3, 1), "gt1ms_n": len(big_g),
                               "gt1ms_ms": round(sum(big_g) / 1e6, 1)}

    # waitB vs staging DMA
    h2di = 0
    dma_tie = 0
    dma_late = []
    b_spins = []
    for w in waits:
        if w["flag"] != "B":
            continue
        b_spins.append(w["end"] - w["start"])
        while h2di + 1 < len(h2d) and h2d[h2di + 1]["end"] <= w["end"]:
            h2di += 1
        if h2di < len(h2d) and h2d[h2di]["start"] < w["end"]:
            if h2d[h2di]["end"] > w["start"]:
                dma_tie += 1
                dma_late.append(h2d[h2di]["end"] - w["start"])
    if b_spins:
        zero = len([s for s in b_spins if s <= 2000])
        print(f"waitB vs staging DMA: {len(b_spins)} waits; spin mean {sum(b_spins)/len(b_spins)/1e3:.1f} us, "
              f"DMA-tied (H2D completes inside the wait): {dma_tie} ({100*dma_tie/len(b_spins):.0f}%), "
              f"near-zero (<=2 us): {zero} ({100*zero/len(b_spins):.0f}%)")
        if dma_late:
            print(f"  DMA completion after waitB start: mean {sum(dma_late)/len(dma_late)/1e3:.1f} us, max {max(dma_late)/1e3:.1f} ms")
        out["waitB"] = {"n": len(b_spins), "mean_us": round(sum(b_spins) / len(b_spins) / 1e3, 1),
                        "dma_tie": dma_tie, "zero_n": zero}

    h2dt = sum(e["end"] - e["start"] for e in h2d)
    h2db = sum(e["bytes"] for e in h2d)
    print(f"staging H2D (>=256 KiB, copy streams): {len(h2d)} copies, {h2db/1e9:.2f} GB, engine-busy {h2dt/1e6:.1f} ms "
          f"-> ~{len(h2d)/max(ntok,1):.2f} blobs/tok, {h2dt/max(len(h2d),1)/1e3:.1f} us/copy mean")

    # per-round
    pr = defaultdict(lambda: {"A": 0.0, "B": 0.0, "C": 0.0, "t": 0.0})
    for w in waits:
        r = pr[w["round"]]
        r[w["flag"]] += w["end"] - w["start"]
        r["t"] += w["end"] - w["start"]
    rr = sorted(pr.items())
    sb = [d0] + samplers + [d1]
    blobr = [len([e for e in h2d if sb[i] <= e["start"] < sb[i + 1]]) for i in range(len(sb) - 1)]
    print(f"\nper-round wait: {len(rr)} rounds, mean {sum(r['t'] for _, r in rr)/max(len(rr),1)/1e6:.2f} ms/round")
    if blobr:
        print(f"staged blobs/round: mean {sum(blobr)/len(blobr):.2f}, max {max(blobr)}, zero-blob rounds "
              f"{len([b for b in blobr if b == 0])}/{len(blobr)}")
    worst = sorted(rr, key=lambda kv: -kv[1]["t"])[:8]
    for ridx, r in worst:
        print(f"  round {ridx:>3d}: total {r['t']/1e6:6.2f} ms  A {r['A']/1e6:6.2f}  B {r['B']/1e6:6.2f}  C {r['C']/1e6:6.2f}")
    out["per_round"] = {"n": len(rr), "mean_ms": round(sum(r["t"] for _, r in rr) / max(len(rr), 1) / 1e6, 2),
                        "blobs_per_round_mean": round(sum(blobr) / max(len(blobr), 1), 2) if blobr else None,
                        "worst": [{"round": i, "ms": round(r["t"] / 1e6, 2), "A": round(r["A"] / 1e6, 2),
                                   "B": round(r["B"] / 1e6, 2), "C": round(r["C"] / 1e6, 2)} for i, r in worst]}

    if js:
        with open(js, "w") as f:
            json.dump(out, f, indent=1)
        print(f"\njson -> {js}")


if __name__ == "__main__":
    main()
