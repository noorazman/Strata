#!/usr/bin/env python3
"""Stage 1.10 — client-side TTFT / decode measurement against a running strata serve.

What this answers: what does the CLIENT (Open WebUI) actually see.  It sends a deterministic
prompt of exactly N target tokens (a prefix of the cached 265k-token corpus), streams
/v1/chat/completions, and timestamps every client-visible stage:

    send -> first byte (SSE headers) -> first SSE data line -> first reasoning delta
          -> first content delta -> completion

plus the per-line arrival times (to compute the steady-state decode rate as the client sees
it, i.e. the last 70 % of visible-token arrivals).  The server's own STRATA_TTFT lines
(engine stages + HTTP stages, one line per request in the server log) join with these on the
wall clock; together they cover the full chain from request arrival to first token at HTTP.

Usage (from anywhere; REPO is absolute):
  s110_client.py --port 8180 --prompt-tokens 4096 --max-new 128 [--thinking on|off] [--label tag]
  s110_client.py --port 8180 --prompt-tokens 100 --max-new 8 --label overhead   # template overhead probe

Each run appends one JSON object to Logs/benchmarks/s110-client.jsonl.
"""
import argparse
import hashlib
import json
import os
import sys
import time
import urllib.request

REPO = "/home/noorazman/dsh/strata/Strata"
sys.path.insert(0, os.path.join(REPO, "tools"))
import strata_tokenizer as _ST  # noqa: E402


def _load_tok():
    tdir = os.path.join(REPO, "packs", "swift-iq3_xxs", "tokenizer")
    vocab = json.loads(open(os.path.join(tdir, "vocab.json"), encoding="utf-8").read())
    tokens = [None] * len(vocab)
    for t, i in vocab.items():
        tokens[i] = t
    merges = open(os.path.join(tdir, "merges.txt"), encoding="utf-8").read().split("\n")
    types = json.loads(open(os.path.join(tdir, "token_type.json")).read())
    return _ST.Tokenizer(tokens, merges, types)


TOK = _load_tok()
CORPUS = os.path.join(REPO, "bench", "v100", "corpus-265k.ids")
OUT = os.path.join(REPO, "Logs", "benchmarks", "s110-client.jsonl")
BASE = ("The quick brown fox jumps over the lazy dog near the riverbank at dawn. "
        "Crimson and gold light spreads across the water as the city slowly wakes up. ")


def load_corpus(max_tokens=265000):
    """The corpus is ONE text encoded in ONE call (the canonical BPE sequence).  Any cut at a
    token position then round-trips exactly (decode(prefix) re-encodes to the prefix): verified
    0/200 random cuts.  (Concatenating per-paragraph encodings is NOT canonical - the tokenizer
    merges the space into the next word (''Paragraph'), so those boundary cuts drifted.)"""
    if os.path.exists(CORPUS):
        with open(CORPUS) as f:
            return [int(x) for x in f.read().split()]
    text = "".join(f"Paragraph {i}. " + BASE * 4 for i in range(2016))
    ids = TOK.encode(text, parse_special=True)[:max_tokens]
    os.makedirs(os.path.dirname(CORPUS), exist_ok=True)
    with open(CORPUS, "w") as f:
        f.write(" ".join(map(str, ids)))
    return ids


def prompt_text(corpus, n):
    """The text that re-encodes to exactly `m <= n` corpus tokens (the cut is verified by a full
    round-trip; on the rare mismatch the cut is stepped down until exact)."""
    m = min(n, len(corpus))
    while m > 0:
        text = TOK.decode(corpus[:m])
        if len(TOK.encode(text, parse_special=True)) == m:
            return text, m
        m -= 1
    raise SystemExit(f"no exact round-trip cut found for target {n}")


def run(port, n_target, max_new, thinking, timeout_s):
    corpus = load_corpus()
    text, n = prompt_text(corpus, n_target)
    body = {
        "model": "swift-iq3_xxs",
        "messages": [{"role": "user", "content": text}],
        "max_completion_tokens": max_new,
        "stream": True,
    }
    if thinking == "off":
        body["chat_template_kwargs"] = {"enable_thinking": False}
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"})
    t_send = time.perf_counter()
    resp = urllib.request.urlopen(req, timeout=timeout_s)
    t_first_byte = time.perf_counter()          # SSE headers are the first bytes on the wire
    t_first_data = t_first_reasoning = t_first_content = None
    arrivals, usage = [], None
    reason_text, content_text = [], []
    while True:
        line = resp.readline()
        if not line:
            break
        t = time.perf_counter()
        if not line.startswith(b"data: "):
            continue
        payload = line[6:].strip()
        if payload == b"[DONE]":
            break
        if t_first_data is None:
            t_first_data = t
        try:
            d = json.loads(payload)
        except ValueError:
            continue
        if d.get("usage"):
            usage = d["usage"]
        ch = (d.get("choices") or [{}])[0]
        delta = ch.get("delta") or {}
        if delta.get("reasoning_content"):
            t_first_reasoning = t
            arrivals.append((t, "r"))
            reason_text.append(delta["reasoning_content"])
        if delta.get("content"):
            t_first_content = t
            arrivals.append((t, "c"))
            content_text.append(delta["content"])
    t_done = time.perf_counter()
    digest = hashlib.md5(("".join(reason_text) + "\x00" + "".join(content_text)).encode()).hexdigest()

    def ms(v):
        return round((v - t_send) * 1000.0, 2) if v is not None else None

    # steady-state decode as the client sees it: mean inter-arrival over the last 70 % of arrivals
    steady = None
    if len(arrivals) >= 20:
        cut = int(len(arrivals) * 0.3)          # steady state = the last 70 % of arrivals
        seg = arrivals[cut:]
        span = seg[-1][0] - seg[0][0]
        if span > 0:
            steady = round((len(seg) - 1) / span, 2)
    out = {
        "t_send": 0.0, "t_first_byte_ms": ms(t_first_byte), "t_first_data_ms": ms(t_first_data),
        "t_first_reasoning_ms": ms(t_first_reasoning), "t_first_content_ms": ms(t_first_content),
        "t_done_ms": ms(t_done), "n_visible_tokens": len(arrivals), "steady_decode_tps": steady,
        "prompt_tokens_target": n_target, "prompt_tokens_actual": n, "thinking": thinking == "on",
        "max_new": max_new, "usage": usage, "text_md5": digest,
        "n_reasoning_chars": len("".join(reason_text)), "n_content_chars": len("".join(content_text)),
    }
    if usage and usage.get("prompt_tokens"):
        out["template_overhead"] = usage["prompt_tokens"] - n
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", type=int, default=8180)
    ap.add_argument("--prompt-tokens", type=int, required=True)
    ap.add_argument("--max-new", type=int, default=128)
    ap.add_argument("--thinking", choices=["on", "off"], default="on")
    ap.add_argument("--label", default="")
    ap.add_argument("--timeout", type=float, default=900.0)
    a = ap.parse_args()
    out = run(a.port, a.prompt_tokens, a.max_new, a.thinking, a.timeout)
    out["label"] = a.label
    out["ts"] = time.strftime("%Y-%m-%d %H:%M:%S")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "a") as f:
        f.write(json.dumps(out) + "\n")
    print(json.dumps(out))


if __name__ == "__main__":
    main()
