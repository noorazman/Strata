#!/usr/bin/env python3
"""Stage 1.10 — summarize one long-context sweep point.

Reads the client jsonl (labels B-<C>-oh / -warm / -r1 / -r2), the engine TTFT jsonl
(Logs/benchmarks/s110-ctx-<C>-engine-ttft.jsonl) and the VRAM CSVs, and prints one table row:

    C  prompt  prefill ms / tok/s  ingest+refill ms  first-window ms  TTFT(ms,client)  decode tps  peak VRAM

plus the determinism verdict and the PLE/experts prefill split.  Used for the stage doc.
"""
import csv
import json
import sys

REPO = "/home/noorazman/dsh/strata/Strata"


def client_rows(c):
    want = {f"B-{c}-oh": "oh", f"B-{c}-warm": "warm", f"B-{c}-r1": "r1", f"B-{c}-r2": "r2"}
    out = {}
    with open(f"{REPO}/Logs/benchmarks/s110-client.jsonl") as f:
        for line in f:
            d = json.loads(line)
            if d.get("label") in want:
                out[want[d["label"]]] = d
    return out


def engine_rows(c):
    path = f"{REPO}/Logs/benchmarks/s110-ctx-{c}-engine-ttft.jsonl"
    rows = []
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line.startswith("strata serve ttft: "):
                    rows.append(json.loads(line[len("strata serve ttft: "):]))
    except FileNotFoundError:
        pass
    return rows


def vram_peak(path):
    try:
        peak = 0
        with open(path) as f:
            for line in csv.reader(f):
                if line and line[0].isdigit():
                    peak = max(peak, int(line[0]))
        return peak
    except (FileNotFoundError, ValueError):
        return None


def main():
    if len(sys.argv) < 2:
        sys.exit("usage: s110_summary.py <max-context> [max-context ...]")
    for c in sys.argv[1:]:
        c = c.strip()
        cl = client_rows(c)
        en = engine_rows(c)
        if not cl:
            print(f"ctx {c}: no client rows yet")
            continue
        r1, r2 = cl.get("r1"), cl.get("r2")
        # match engine rows to client runs by prompt size order: r1 then r2
        e = {
            "prefill_ms": None, "prefill_tps": None, "ingest_ms": None, "refill_ms": None,
            "first_window_ms": None, "first_tok_out_lag_us": None,
            "ple_ms": None, "experts_streamed": None, "experts_resident": None, "chunks": None,
        }
        n1 = (r1 or {}).get("usage", {}).get("prompt_tokens")
        for row in en:
            if row.get("max_new") == (r1 or {}).get("max_new") and row.get("prompt_tokens") == n1:
                # the FIRST matching row is r1
                if e["prefill_ms"] is None:
                    # wall time of sp.run (the *_us fields are microseconds).  PrefillStats.ms_total
                    # accumulates over ALL requests of this server (probe+warmup+r1+r2), so the
                    # per-request rate below uses the wall span; PLE stays the cumulative percentage.
                    e["prefill_ms"] = (row["t_prefill_end_us"] - row["t_prefill_start_us"]) / 1000.0
                    e["ingest_ms"] = row["t_prefill_start_us"] / 1000.0
                    e["refill_ms"] = (row["t_ready_us"] - row["t_prefill_end_us"]) / 1000.0
                    e["first_window_ms"] = (row["t_first_token_us"] - row["t_first_window_us"]) / 1000.0
                    e["first_tok_out_lag_us"] = row["t_first_token_out_us"] - row["t_first_token_us"]
                    e["ple_ms"] = row.get("prefill_ple_ms")
                    e["experts_streamed"] = row.get("experts_streamed")
                    e["experts_resident"] = row.get("experts_resident")
                    e["chunks"] = row.get("prefill_chunks")
        n = (r1 or {}).get("prompt_tokens_actual", 0)
        pf_tps = round(n / (e["prefill_ms"] / 1000.0), 1) if e["prefill_ms"] else None
        v1 = vram_peak(f"{REPO}/Logs/benchmarks/s110-ctx-{c}-vram1.csv")
        v2 = vram_peak(f"{REPO}/Logs/benchmarks/s110-ctx-{c}-vram2.csv")
        det = None
        if r1 and r2:
            det = "IDENTICAL" if r1.get("text_md5") == r2.get("text_md5") else "DIVERGED"
        print(f"--- ctx {c}")
        print(f"  prompt={n}  prefill={e['prefill_ms']:.0f} ms ({pf_tps} tok/s, {e['chunks']} chunks, "
              f"PLE {e['ple_ms']:.0f} ms, experts streamed/resident {e['experts_streamed']}/{e['experts_resident']})")
        print(f"  ingest={e['ingest_ms']:.3f} ms  refill={e['refill_ms']:.3f} ms  "
              f"first-window={e['first_window_ms']:.3f} ms  T-line lag={e['first_tok_out_lag_us']} us")
        for tag, r in (("r1", r1), ("r2", r2)):
            if r:
                ttft = r.get("t_first_reasoning_ms") or r.get("t_first_content_ms")
                print(f"  {tag}: client TTFT={ttft} ms  steady decode={r.get('steady_decode_tps')} tok/s  "
                      f"done={r.get('t_done_ms')} ms  md5={r.get('text_md5', '')[:12]}")
        print(f"  determinism: {det}   peak VRAM: r1={v1} MiB r2={v2} MiB")


if __name__ == "__main__":
    main()
