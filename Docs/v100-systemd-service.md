# Strata V100 — systemd Service (production inference server)

Persistent `systemd` service that runs the current Stage 1.4 Strata build as a
continuous OpenAI/Anthropic inference API on the 32 GB V100 (GPU0). This is a
deployment change only — no inference algorithms, kernels, scheduling, PLE,
PCIe behavior, or performance settings were modified.

## Service file

- **Installed unit:** `/etc/systemd/system/strata.service`
- **Source copy (tracked):** `Docs/strata.service` (in this repo)

To (re)install after editing:

```bash
sudo cp Docs/strata.service /etc/systemd/system/strata.service
sudo systemctl daemon-reload
```

## Exact commands

```bash
sudo systemctl start strata      # start
sudo systemctl stop strata       # stop (clean; SIGINT → QUIT to the engine)
sudo systemctl restart strata    # restart
sudo systemctl status strata     # status
journalctl -u strata -f          # follow the logs
journalctl -u strata -n 50       # last 50 lines
```

Enable on boot (already done):

```bash
sudo systemctl enable strata     # wanted by multi-user.target
```

## API endpoint

- **Base (local):** `http://127.0.0.1:8180/v1`
- **Base (LAN, from other PCs):** `http://192.168.0.33:8180/v1`
  (the unit binds `0.0.0.0`, so the API is reachable on all interfaces;
  use this machine's current LAN IP — `192.168.0.33` as of 2026-09-27)
- **Model alias (the `model` field in requests):** `swift-iq3_xxs`
- **OpenAI:** `POST /v1/chat/completions` (stream and non-stream)
- **Anthropic:** `POST /v1/messages` (stream and non-stream)
- **Models:** `GET /v1/models`
- **Health:** `GET /health`
- **Web UI:** `http://127.0.0.1:8180/`

Example request:

```bash
curl -s -X POST http://127.0.0.1:8180/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"swift-iq3_xxs","messages":[{"role":"user","content":"Hello"}],
       "max_tokens":64,"stream":false}'
```

The model is a thinking model: short `max_tokens` may return only
`reasoning_content`; raise `max_tokens` to get the final `content`.

**Security note:** the API currently has **no API key** (`/health` shows
`"api_key": false`), so any host that can reach `192.168.0.33:8180` can call
it. To add one: append `--api-key <key>` to `ExecStart` (or set
`Environment=STRATA_API_KEY=<key>`), `sudo systemctl daemon-reload`,
`sudo systemctl restart strata`; clients then send
`Authorization: Bearer <key>` (OpenAI) or `x-api-key: <key>` (Anthropic).

## Model / configuration

- **Model:** Swift-1.5-Qwen3.8-Flash-Next (IQ3_XXS, 2 shards, 70.74 GiB)
  at `/mnt/ssd/llm_models/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF/`
- **Engine config:** `strata-swift-iq3_xxs.json` (repo root, gitignored)
- **API port:** `8180`
- **Context:** `32768`
- **KV:** `int8`

## GPU selection

- **Primary production/test GPU:** the 32 GB V100 = **GPU0**
- **Explicit selection:** `CUDA_VISIBLE_DEVICES=0` in the unit (not relying on
  accidental device ordering)
- The 16 GB V100 (GPU4) remains the validation GPU and is untouched.

## Current launch flags (known-good V100 settings)

From `strata-swift-iq3_xxs.json` (the Stage 1.4 known-good baseline):

```text
--pack packs/swift-iq3_xxs
--native .../Swift-...-IQ3_XXS-00001-of-00002.gguf
--ple-gguf .../Swift-...-IQ3_XXS-00001-of-00002.gguf
--expert-profile data/expert-profile.bin
--expert-cache auto
--prefill 2048
--spec 4 --spec-min-p 0.5
--pool-workers 24
--mtp mtp/rt
--max-context 32768
--kv int8
```

- `--ple-io ram` is the **default** PLE storage mode (Stage 1.2B); no flag is
  needed. The ~26.82 GiB PLE table stays **RAM-resident**. There is **no**
  automatic fallback to `direct`/`mmap` — if RAM is insufficient the run fails
  clearly.
- `--pool-workers 24` (Stage 1.1) and `--pcie-frac 0.2` (Stage 1.3 default)
  are the known-good settings.

## Environment preserved by the unit

- `CUDA_VISIBLE_DEVICES=0` — explicit 32 GB V100 selection
- `LD_LIBRARY_PATH=/usr/local/cuda/lib64` — CUDA runtime libraries
- `LimitMEMLOCK=infinity` — lifts systemd's 8 MB `LimitMEMLOCK` so the
  ~40 GiB expert arena can pin host memory via `mlock` (the equivalent of the
  old `ulimit -l unlimited`)
- Runs as the normal user **`noorazman`** (not root)

## RAM

- The machine has 128 GB RAM.
- `--ple-io ram` keeps the ~26.82 GiB PLE table resident in RAM (intended).
- Peak process RSS is ~67 GiB (PLE table + expert arena + KV + overhead),
  leaving ~44 GiB headroom.

## Validation (performed)

All of the following were verified through `systemctl`:

1. Started through `systemctl`. ✓
2. Process running. ✓
3. API port 8180 listening. ✓
4. Real inference request through the API. ✓
5. Streaming works. ✓
6. Output correct (e.g. "Paris", "Berlin"). ✓
7. Uses the 32 GB V100 (GPU0, ~19 GiB VRAM). ✓
8. PLE RAM-resident (`PLE I/O mode: ram`, table preloaded). ✓
9. No unexpected NVMe PLE reads during inference (diskstats delta = 0). ✓
10. No CUDA errors. ✓
11. Stopped cleanly. ✓
12. Started again. ✓
13. Remains functional after restart. ✓

## Troubleshooting

- **Service fails to start / port already in use:**
  `ss -ltnp | grep 8180` — if a stray `strata`/`serve.server` is running,
  `sudo systemctl stop strata` and kill leftovers, then `sudo systemctl start strata`.
- **Check logs:** `journalctl -u strata -f` (server + startup) and the engine
  log `strata-swift-iq3_xxs.log` (PLE/arena/GPU detail).
- **GPU not selected / wrong card:** confirm `CUDA_VISIBLE_DEVICES=0` in the
  unit and `nvidia-smi` shows GPU0 (32 GB) in use.
- **PLE not RAM-resident / high NVMe reads:** confirm `--ple-io ram` is the
  default (no flag) and that enough RAM is free; the engine fails clearly
  rather than silently switching modes.
- **Arena not pinned (`mlock failed` in the log):** confirm
  `LimitMEMLOCK=infinity` is present in the unit and `sudo systemctl daemon-reload`
  was run after editing.
- **Crash loop:** `sudo systemctl status strata` shows the exit code;
  `Restart=on-failure` auto-restarts after 5 s. Read the journal for the error.
