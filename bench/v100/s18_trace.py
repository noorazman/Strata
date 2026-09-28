#!/usr/bin/env python3
"""Stage 1.8 — QSA attention-path breakdown of an nsys sqlite capture
(cuda-graph-trace=node).

Method: kernels on the main (verify-window) stream run in recorded order and
follow a fixed per-layer pattern.  Every `kv_append*` kernel is the K/V cache
append of one QSA layer for one token group; the mixer around it is decoded
with fixed offsets (verified against the actual stream order):

  back:  [copy_from_mapped?] apply (K rope) norm (K norm) V-mmVQ K-mmVQ
         idxk-mmvf quantize <gr_read tail>
  kv_append
  fwd:   append (indexer) Q-mmVQ norm (Q norm) apply (Q rope) idxq-mmvf
         norm (idxq norm) apply (idxq rope)
         <select: block_scores+block_topk (fast) | per-token qsa_index+topk>
         <attn: attn_chunk+attn_merge (fast) | per-token kv_gather+qsa_attend>
         gate quantize O-mmVQ <next layer's gr_read>

Everything between one layer's O projection and the next QSA layer's gr_read
(GDN mixer, MoE route/experts, GR, post waits, shared expert) is reported as
the non-QSA remainder.  The MTP draft runs on its own stream and is reported
as a separate block.

Usage: s18_trace.py <capture.sqlite> [decode_tokens]
"""
import re
import sqlite3
import sys
from collections import defaultdict


def load(path):
    db = sqlite3.connect(path)
    db.row_factory = sqlite3.Row
    cur = db.cursor()
    k = [dict(r) for r in cur.execute(
        "SELECT start, end, streamId, s.value AS name, gridId, graphNodeId,"
        " gridX, gridY, gridZ, blockX, blockY, blockZ, registersPerThread,"
        " staticSharedMemory, dynamicSharedMemory, sharedMemoryExecuted,"
        " localMemoryPerThread FROM CUPTI_ACTIVITY_KIND_KERNEL k LEFT JOIN StringIds s ON s.id = k.demangledName"
        " ORDER BY start")]
    m = [dict(r) for r in cur.execute(
        "SELECT start, end, streamId, bytes, copyKind FROM CUPTI_ACTIVITY_KIND_MEMCPY ORDER BY start")]
    return k, m, db


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


def base(name):
    if not name:
        return "<null>"
    name = re.sub(r"^void\s+", "", name)
    m = re.search(r"([a-zA-Z_][a-zA-Z0-9_]*)(<|\(|\s)", name)
    return m.group(1) if m else name


def is_mmvq(b):
    return bool(re.search(r"mmvq(_multi)?_kernel$", b))


def busy(events):
    ev = sorted(events)
    if not ev:
        return 0
    tot, cs, ce = 0, ev[0][0], ev[0][1]
    for s, e in ev[1:]:
        if s > ce:
            tot += ce - cs
            cs, ce = s, e
        else:
            ce = max(ce, e)
    return tot + ce - cs


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    k, m, db = load(sys.argv[1])
    ntok = int(sys.argv[2]) if len(sys.argv) > 2 else 256
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
    mtp_stream = None
    for e in dec:
        if "mtp" in e["b"]:
            mtp_stream = e["streamId"]
            break
    all_ev = [(e["start"], e["end"]) for e in dec]
    main_ev = [(e["start"], e["end"]) for e in dec if e["streamId"] == main_stream]
    mtp_ev = [(e["start"], e["end"]) for e in dec if mtp_stream and e["streamId"] == mtp_stream]
    print(f"decode window: {span/1e6:.1f} ms, {len(dec)} kernels; main stream {main_stream} "
          f"({scnt.get(main_stream, 0)}), MTP stream {mtp_stream} ({scnt.get(mtp_stream, 0) if mtp_stream else 0})")
    print(f"GPU busy union {busy(all_ev)/1e6:.1f} ms = {100*busy(all_ev)/span:.1f}% of span; "
          f"main {busy(main_ev)/1e6:.1f} ms, MTP {busy(mtp_ev)/1e6:.1f} ms")

    # ---- full per-kernel table
    agg = defaultdict(lambda: [0, 0, None])
    for e in dec:
        a = agg[e["name"]]
        a[0] += 1
        a[1] += e["end"] - e["start"]
        if a[2] is None:
            a[2] = (e["gridX"], e["gridY"], e["gridZ"], e["blockX"],
                    e["registersPerThread"],
                    (e["staticSharedMemory"] + e["dynamicSharedMemory"]) or e["sharedMemoryExecuted"],
                    e["localMemoryPerThread"])
    rows = sorted(agg.items(), key=lambda kv: -kv[1][1])
    print(f"\n== per-kernel top 30 (all streams, decode) ==")
    print(f"{'kernel':60s} {'n':>7s} {'ms':>8s} {'mean us':>8s}  grid/block r=regs sm=smem")
    for name, (n, t, cfg) in rows[:30]:
        gx, gy, gz, bx, regs, smem, lmem = cfg
        grid = f"{gx}x{gy}x{gz}" if (gy > 1 or gz > 1) else str(gx)
        print(f"{name[:60]:60s} {n:>7d} {t/1e6:>8.1f} {t/n/1e3:>8.2f}  {grid:>10s}/{bx:<4d} r={regs:<4d} sm={smem}")

    # ---- QSA segment decoder on the main stream
    import os
    DBG = os.environ.get("S18_DEBUG_BUCKETS") == "1"
    comp = defaultdict(lambda: [0.0, 0])
    compk = defaultdict(lambda: defaultdict(lambda: [0.0, 0]))
    nonqsa = [0.0, 0]
    seq = sorted([e for e in dec if e["streamId"] == main_stream], key=lambda e: e["start"])
    n = len(seq)

    def put(cname, e):
        comp[cname][0] += e["end"] - e["start"]
        comp[cname][1] += 1

    KVAPP = ("kv_append_kernel", "kv_append_q8_kernel")
    i = 0
    nseg = 0
    while i < n:
        b = seq[i]["b"]
        if b in KVAPP:
            j = i
            nseg += 1
            # --- backward segment (K/V projections + indexer K + norm + rope)
            mmvq_back = []
            t = i - 1
            while t >= 0 and i - t <= 30:
                bt = seq[t]["b"]
                if bt in ("gr_up_multi_kernel", "gr_up_kernel"):
                    break
                if is_mmvq(bt):
                    mmvq_back.append(t)
                t -= 1
            # walking backwards, the per-token pattern is
            # [..., idxk_t, K_t, V_t, norm_t, rope_t] so mmvq_back is
            # [V_last, K_last, V_prev, K_prev, ...]
            for m, idx in enumerate(mmvq_back):
                put(("projection V (GEMV)" if m % 2 == 0 else
                     "projection K (GEMV)"), seq[idx])
            for t in range(t + 1, i):
                bt = seq[t]["b"]
                if t in mmvq_back:
                    continue
                if bt == "bf16_f32_mmvf_kernel":
                    put("indexer K projection (bf16 mmvf)", seq[t])
                elif bt == "norm":
                    put("K norm (rms)", seq[t])
                elif bt == "apply":
                    put("K rope", seq[t])
                elif bt in ("native_quantize_q8_1_kernel", "quantize_q8_1_kernel"):
                    put("activation quantize (q8_1)", seq[t])
                elif bt in ("copy_from_mapped_kernel", "copy_i32_from_mapped_kernel"):
                    put("step/pos staging (mapped copy)", seq[t])
                else:
                    put("QSA pre other", seq[t])
            put("KV cache append", seq[i])
            j = i + 1
            # --- forward: indexer append, Q projection, norms/ropes, selection
            while j < n:
                bt = seq[j]["b"]
                if bt in ("block_scores_kernel", "qsa_index_kernel"):
                    break
                if bt == "append":
                    put("indexer key append", seq[j])
                elif bt in KVAPP:
                    put("KV cache append", seq[j])
                elif is_mmvq(bt):
                    put("projection Q (GEMV)", seq[j])
                elif bt == "norm":
                    put("Q/idxq norm (rms)", seq[j])
                elif bt == "apply":
                    put("Q/idxq rope", seq[j])
                elif bt == "bf16_f32_mmvf_kernel":
                    put("indexer Q projection (bf16 mmvf)", seq[j])
                elif bt in ("copy_from_mapped_kernel", "copy_i32_from_mapped_kernel"):
                    put("step/pos staging (mapped copy)", seq[j])
                else:
                    put("QSA pre other", seq[j])
                j += 1
                if j - i > 30:
                    break
            # --- selection
            sel_slow = 0
            sel_fast = 0
            while j < n:
                bt = seq[j]["b"]
                if bt == "block_scores_kernel":
                    put("selection block scores (fast)", seq[j])
                    sel_fast += 1
                    j += 1
                elif bt == "block_topk_kernel":
                    put("selection block topk (fast)", seq[j])
                    j += 1
                elif bt == "qsa_index_kernel":
                    put("selection cell scores (slow)", seq[j])
                    sel_slow += 1
                    j += 1
                elif bt == "topk_kernel":
                    put("selection cell topk (slow)", seq[j])
                    j += 1
                else:
                    break
            # --- attention
            at_slow = 0
            while j < n:
                bt = seq[j]["b"]
                if bt == "attn_chunk_kernel":
                    put("attention chunk (fast)", seq[j])
                    j += 1
                elif bt == "attn_merge_kernel":
                    put("attention merge (fast)", seq[j])
                    j += 1
                elif bt in ("kv_gather_kernel", "kv_gather_q8_kernel"):
                    put("KV gather (slow)", seq[j])
                    at_slow += 1
                    j += 1
                elif bt == "qsa_attend_kernel":
                    put("attention attend (slow)", seq[j])
                    j += 1
                else:
                    break
            # --- gate + output quantize + O projection (until next gr_read)
            o_seen = False
            while j < n:
                bt = seq[j]["b"]
                if bt in ("gr_norm_multi_kernel", "gr_norm_kernel"):
                    break
                if bt == "gate":
                    put("gate (sigmoid*attn)", seq[j])
                elif is_mmvq(bt) and not o_seen:
                    put("projection O (GEMV)", seq[j])
                    o_seen = True
                elif bt in ("native_quantize_q8_1_kernel", "quantize_q8_1_kernel") and not o_seen:
                    put("attn output quantize (q8_1)", seq[j])
                elif bt in ("native_quantize_q8_1_kernel", "quantize_q8_1_kernel"):
                    put("attn output quantize (q8_1)", seq[j])
                else:
                    put("QSA epilogue other", seq[j])
                j += 1
                if j - i > 60:
                    break
            i = j
        else:
            nonqsa[0] += seq[i]["end"] - seq[i]["start"]
            nonqsa[1] += 1
            i += 1

    grand = 0.0
    print(f"\n== QSA attention path (main model, {ntok} tokens, {nseg} QSA segments) ==")
    print(f"{'component':38s} {'us/tok':>9s} {'ms total':>9s} {'calls':>7s} {'calls/tok':>9s} {'mean us':>8s}")
    for cname, (t, c) in sorted(comp.items(), key=lambda kv: -kv[1][0]):
        if t <= 0:
            continue
        grand += t
        print(f"{cname:38s} {t/1e3/ntok:>9.2f} {t/1e6:>9.1f} {c:>7d} {c/ntok:>9.2f} {t/c/1e3:>8.2f}")
    print(f"{'-- QSA attention path subtotal':38s} {grand/1e3/ntok:>9.2f} {grand/1e6:>9.1f}")
    print(f"{'non-QSA main stream (GDN/MoE/GR/post/head)':38s} "
          f"{nonqsa[0]/1e3/ntok:>9.2f} {nonqsa[0]/1e6:>9.1f} {nonqsa[1]:>7d}")
    print(f"span/token {span/1e6/ntok:.3f} ms; QSA path = {100*grand/span:.1f}% of span, "
          f"{100*grand/(grand+nonqsa[0]):.1f}% of main-stream kernel time; "
          f"QSA kernels/token {sum(c for _, c in comp.values())/ntok:.1f} + "
          f"non-QSA {nonqsa[1]/ntok:.0f} + MTP")

    # ---- MTP block
    if mtp_stream:
        mtpk = [e for e in dec if e["streamId"] == mtp_stream]
        mt = sum(e["end"] - e["start"] for e in mtpk)
        print(f"\n== MTP draft stream: {len(mtpk)} kernels, {mt/1e6:.1f} ms "
              f"({mt/1e3/ntok:.2f} us/tok) ==")
        magg = defaultdict(lambda: [0, 0])
        for e in mtpk:
            a = magg[e["b"]]
            a[0] += 1
            a[1] += e["end"] - e["start"]
        for name, (c, t) in sorted(magg.items(), key=lambda kv: -kv[1][1])[:14]:
            print(f"  {name:40s} {c:>6d} {t/1e6:>8.1f} ms {t/c/1e3:>7.2f} us")

    # ---- memcpy on the main stream
    mm = [e for e in mem if e["streamId"] == main_stream]
    mb = sum(e["bytes"] for e in mm)
    mms = sum(e["end"] - e["start"] for e in mm)
    print(f"\n== memcpy main stream: {len(mm)} ops, {mb/1e6:.1f} MB, {mms/1e6:.1f} ms "
          f"({mms/1e3/ntok:.2f} us/tok) ==")
    sz = defaultdict(lambda: [0, 0])
    for e in mm:
        key = round(e["bytes"] / 1024)
        sz[key][0] += 1
        sz[key][1] += e["end"] - e["start"]
    for kb, (c, t) in sorted(sz.items(), key=lambda kv: -kv[1][1])[:10]:
        print(f"  ~{kb:>6d} KiB: {c:>6d} ops {t/1e6:>7.1f} ms")


if __name__ == "__main__":
    main()
