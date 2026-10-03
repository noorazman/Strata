# Release 2 — Multi-GPU MoE on 2× V100 32GB: resume document

Paused: 2026-10-03. This doc is the complete hand-off for resuming in a fresh session. It replaces
`Docs/v100-release2-phase2-resume.md` (the older phase-2 mission — move QSA state to an auxiliary
device — which was explicitly put aside by the user; do not use it as a reference). No engine code
has been changed yet; everything below is survey + state.

## 0. Status (2026-10-03, pre-reboot)

- **3.1 DONE.** On `release2-v100-multigpu` @ `22fec65`, pushed.
- **3.2 DONE (pending post-reboot retest).** Probe seed bug fixed (`measure_p2p_gbps`
  `cudaMemset` now runs on the source device, before `cudaSetDevice(dst)`). `device_plan_report`
  display bug fixed (L390 `* 100` → `* 1000`, so 1.31 GB/s no longer reads "131 MB/s"). Measured
  **true bidirectional P2P: 0→1 = 1.31 GB/s, 1→0 = 1.52 GB/s** (peer access enabled before
  measuring — genuine root-complex P2P, not host staging). Recorded in
  `Docs/v100-release2-multigpu.md`. Committed + pushed as `1dbf5d1`.
- **Driver wedge:** after the probe, a 10:56 hardware storm set **Xid 31 (MMU fault) then Xid 154
  "Node Reboot Required" on all 5 cards**; every non-reboot reset path is blocked without root
  (`nvidia-persistenced` re-enables PM within ~100 ms of `-pm 0`; `systemctl stop` needs polkit;
  `sudo`/`pkexec` interactive). A **node reboot is required** to clear it.
- **Reboot cost is low:** the only non-idle tenant besides strata cards 0/1 is GPU 4, holding the
  user's own `llama-server` (PID 1697779, ~15.5 GB, port 8081) under
  `ik-llama-hermesQ3-xxs-gpu4-swift.service` — **enabled**, so it auto-restarts after reboot.
  `deepseek-harness.service` also auto-restarts. `hindsight-api` (PID 1787) only holds device fds.
- **3.3 surveyed** (see §4): the hand-off today stages through mapped pinned memory
  (`cudaHostAlloc(cudaHostAllocMapped|Portable)`, generate.cpp L4315–4367); a direct P2P rewrite
  allocates the hand-off buffer on the source device so the destination reads it peer-addressable.
  The `copy_from_mapped` kernel already reads through any device-visible pointer, so only the
  allocation changes; ordering is host-ordered by `cudaStreamSynchronize(cs_)` (verify.cpp L1336).
- **Next after reboot:** verify `nvidia-smi -q` shows `GPU Recovery Action: None` on cards 0/1 (and
  all cards), then re-run `timeout 120 env CUDA_VISIBLE_DEVICES=0,1 ./build-sm70/strata-device
  --devices 0,1 --p2p` to confirm 1.31/1.52 GB/s and a **clean exit** (settles whether the original
  post-`main` exit hang was residual stuck-PID state — likely, given the Xid 31 faults). Then 3.3
  (implement direct P2P hand-off), 3.4 (residency), 3.5 (ctest), 3.6 (2-GPU correctness + benchmarks),
  3.7 (final doc + commit + push + notify).

## 1. Mission

Complete **Release 2 — Multi-GPU MoE on 2× Tesla V100 32GB (GPU 0 & GPU 1) with Direct Hardware P2P
& 100% Expert VRAM Residency** in `~/dsh/strata/Strata` (baseline commit `3185b12` on
`release3-qwen35-dense`, target branch `release2-v100-multigpu`).

Core mandate:

- 48 MoE layers split evenly: layers 0–23 on GPU 0, layers 24–47 on GPU 1
  (`--layer-split 24` or `--layer-split auto`).
- Each card runs an independent ExpertCache: 24 layers × 512 experts = 12,288 experts/card,
  ~19.4 GiB. 100% expert residency on both cards.
- PLE 26.82 GiB table stays in host RAM (`--ple-io ram` is the default).
- The remaining ~10–12 GiB per card goes to the int8 KV cache.
- Stage hand-off goes over **direct `cudaMemcpyPeerAsync`** (hardware P2P), no host bounce;
  pinned host staging stays as the fallback path.

Steps:

- **3.1** git branch transition: merge `release3-qwen35-dense` into `main`, push, create
  `release2-v100-multigpu` from it, `git push -u origin release2-v100-multigpu`.
- **3.2** fix the `strata-device` P2P probe, measure bidirectional GB/s, record in
  `Docs/v100-release2-multigpu.md`.
- **3.3** wire direct P2P stage hand-off in `src/program/generate.cpp` / `src/core/layer.cpp`:
  check `DevicePlan::p2p[src][dst]==1`, use `cudaMemcpyPeerAsync` + CUDA events, keep the pinned
  host staging fallback, log `strata serve: layer split hand-off: direct P2P (cudaMemcpyPeerAsync)`
  at startup (and the fallback line when it is not used).
- **3.4** verify 100% expert residency (12,288/card), PLE in RAM, int8 KV filling the remainder.
- **3.5** clean SM70 build + ctest `-R "parity|selftest|hostref"` — expect 30/31.
- **3.6** 2-GPU correctness run (first token 271, golden 32-prefix, balanced VRAM, 0 CUDA errors) +
  16K/32K benchmarks vs Strata-Pure.
- **3.7** write `Docs/v100-release2-multigpu.md` (P2P bandwidth + benchmarks + architecture),
  commit, `git push -u origin release2-v100-multigpu`,
  `~/.dsh/bin/dsh-notify "Release 2: 2x V100 32GB multi-GPU P2P setup and benchmarks complete"`,
  final structured summary vs Strata-Pure and Release 1.

## 2. State at pause

- On branch `release3-qwen35-dense` at `3185b12` ("Docs: Stage 3.0 sync record + Release 3 benchmarks").
  No local `main` branch (only `origin/main`, `local-mirror/main`, `upstream/main`).
  `origin/main` last known at `6da1f66` (v0.1.2) — the merge in step 3.1 will be large.
- Working tree: modified `Logs/benchmarks/*` (s110-* engine/ttft/vram logs) + many untracked
  `Logs/*` and `bench/v100/*` artifacts + this untracked doc. No modified sources.
- `build-sm70/` exists; `build-sm70/strata` built 2026-10-03 04:46, `build-sm70/strata-device`
  built 2026-10-03 00:04.
- GPUs 0 & 1 free (4 MiB each), `strata.service` **inactive**.
- GPU 4 holds ~15.5 GB (some other tenant) — leave GPUs 2–4 alone.

## 3. Machine + environment facts

- 5 GPUs: 0 = V100-PCIE-32GB (strata.service when running), 1 = V100-PCIE-32GB (the second
  Release-2 card), 2–4 = V100-SXM2-16GB. No NVLink anywhere; P2P is PCIe root-complex
  (`cudaDeviceCanAccessPeer` true both directions for 0↔1).
- NUMA: GPU 0/1 on node 0.
- Model: Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS (48 MoE layers, 512 experts/layer).
- Build: `.venv/bin/cmake --build build-sm70 -j 40`.
- Tests: `cd build-sm70 && CUDA_VISIBLE_DEVICES=0 ctest --output-on-failure -R "parity|selftest|hostref"`
  → expect 30/31 pass.
- Determinism anchors: first token must be **271**; golden 32-prefix check via
  `bench/v100/s18check.py --prefix <json> 32`; 256-token md5 `cdb7f7d056f339ba704d3bb9620a1dec`
  at 16K/32K int8 KV. Correctness run expected output starts with tokens 9419, 11, 821, …
- Benchmark harness: `bench/v100/s110.sh` (ctx 16384 / ctx 32768) + `bench/v100/bench.py`
  (writes `Logs/benchmarks/<label>.json`). Comparison target: Strata-Pure at
  `/home/noorazman/Strata-Pure`. Metrics to capture: TTFT, prefill tok/s, decode tok/s,
  spec acceptance, VRAM peaks, P2P GB/s.
- Run the 2-GPU serve with `CUDA_VISIBLE_DEVICES=0,1` and `--layer-split auto` (or `24`).

## 4. Code survey — the layer-split hand-off (the P2P wiring point)

The hand-off mechanism already exists and works over mapped pinned memory. Key locations:

- **`src/program/generate.cpp` L4305–4367 — the hand-off wiring (step 3.3 goes here).**
  Comment L4305: *"the hand-offs are mapped pinned memory, portable: a stage on another GPU
  reads it."* A `std::vector<float*> hand` of `n_stages - 1` buffers, each allocated
  `cudaHostAlloc(cudaHostAllocMapped | cudaHostAllocPortable)` + `cudaHostGetDevicePointer`
  (L4319–4327), sized `kVerifyMaxT * Verifier::handoff_floats(g) * sizeof(float)` (L4316–4317).
  `set_stage(...)` per stage (L4331–4332) with `hand[st-1]` in / `hand[st]` out;
  `set_next(&stage_ver(st+1), &split_drive)` (L4361). Startup log line L4366:
  `strata serve: layer split: layers %s, one hand-off per window`.
- **`include/strata/core/verify.hpp` L107–121 — the contract.** `set_stage(layer_begin,
  layer_end, handoff_in, handoff_out)`; `handoff_floats(g) = hc*n_embd + n_embd + hc` floats per
  token (residual R + pending write bo + inject). L112: *"Both pointers must be device-visible
  (mapped pinned memory, portable when the stages are on two devices)."* — the contract a direct
  P2P device buffer must also satisfy.
- **`src/core/verify.cpp` — where the hand-off is consumed.**
  - L522–527: a later stage (lb_ > 0) reads `hand_in_` per token via `copy_from_mapped`
    (R, bo, inj2).
  - L955–957: the earlier stage writes `hand_out_` per token via `copy_from_mapped`
    (R, bo, inj2).
  - L1336: the stage `cudaStreamSynchronize(cs_)`s its own window before L1383–1385, where
    `le_ < g.n_layers` stages call `next_->run(...)`. **The hand-off ordering is host-ordered
    today by this sync** — a direct-P2P rewrite must preserve that ordering (events or the
    existing sync).
- **`src/kernels/cuda/elementwise.cu` L226 / L264–273 — `copy_from_mapped`** is a plain
  `float4` kernel `copy_from_mapped_kernel(float4* dst, const volatile float4* src, int64_t n4)`
  — it reads through whatever device pointer `src` is, so it already works on a
  peer-addressable device buffer; only the *allocation* of the hand-off buffer changes, not the
  copy kernel.
- **`src/program/generate.cpp` L712–730 — `struct GpuStage`**: `dev`, `lb`/`le`, `wt`, `ss`,
  `cache` (per-stage ExpertCache), `ver`, `stream`, `d_res`. Each stage loads its own weights
  arena (L2273–2280), its own session (sized by its layer range, allocated after the split
  search), its own expert cache (L2996–3024: `st.cache.open_sized/open`, admitted from the
  profile, `fill_slot_blocking`), per-stage PCIe probe (L2313–2318), and the last stage holds
  the head + drafter (L4293–4297, `hist_dev` = last stage's device).
- **`src/program/generate.cpp` L1294–1338 — `--layer-split` parsing/validation**: K or
  "K1,K2,…" or `auto` (auto → devices 1..n-1, L1317); `split_same` (single-GPU A/B mode);
  `multi_gpu` flag; `--layer-split` requires `--serve` mode (L1314) and rising K ≥ 2.
  `--split-skip-if-fits` clears the split when CUDA0 fits everything (L2240–2251).
- **`src/program/generate.cpp` L1900–1926 — the DevicePlan is already built before the first
  allocation** (step 3.3 consumes it): `make_device_plan(devices_spec, /*probe_bandwidth=*/false)`
  with `--devices` / `$STRATA_DEVICES` (default "0"), `cudaSetDevice(primary)`, startup report
  line via `device_plan_report`.
- **`include/strata/core/devices.hpp` + `src/core/device.cu` — the plan infrastructure.**
  `DevicePlan { ordinals, info, p2p[i][j], p2p_gbps[i][j] }`; `p2p[i][j]==1` when device
  `ordinals[j]`'s memory is directly addressable from `ordinals[i]` (peer access actually
  enabled at plan time, not merely `cudaDeviceCanAccessPeer`). `make_device_plan` enables peer
  access on every usable pair (device.cu L346–364); `probe_bandwidth` measures real D2D GB/s
  per enabled pair.
- **`src/core/layer.cpp` L382** — doorbell handoff copies (`moe_route` D2D) — inspect for the
  stage-1 hand-off path if needed; the `h_handoff_` name was not found (the buffer is the
  `hand` vector above).

## 5. Step 3.2 — the P2P probe bug (root-caused, fix not yet applied)

`strata-device --devices 0,1 --p2p --selftest` fails with "invalid argument" at
`src/core/device.cu:307` (`p2p probe: seed`).

Root cause in `measure_p2p_gbps` (device.cu L282–326): the active device is `dst` (set L303)
when the source buffer `g.s` (allocated on `src`, L302) is seeded —

```
301  check(cudaSetDevice(src), "cudaSetDevice(src)");
302  check(cudaMalloc(&g.s, bytes), "p2p probe: cudaMalloc on the source");
303  check(cudaSetDevice(dst), "cudaSetDevice(dst)");
304  check(cudaMalloc(&g.d, bytes), "p2p probe: cudaMalloc on the destination");
307  check(cudaMemset(g.s, 0x5a, bytes), "p2p probe: seed");          // active dev = dst, buffer on src
308  check(cudaSetDevice(src), "cudaSetDevice(src) again");
```

`cudaMemset` (synchronous runtime API) requires the pointer to live on the active device,
so L307 faults with "invalid argument". **Fix: insert
`check(cudaSetDevice(src), "cudaSetDevice(src) for seed");` immediately before L307** (or
move the L308 call up). The rest of the probe (events on the source device,
`cudaMemcpyPeer` warmup + timed loop, 250 ms / 1 GiB cap) is sound; `~Guard` cleanup and the
self-test in `src/core/device_main.cpp` (`run_cross_device_selftest`, L49+ — 1 MiB pattern,
`cudaMemcpyPeer` when p2p else host staging, bit-for-bit verify) are correct.

After the fix: rebuild, run
`CUDA_VISIBLE_DEVICES=0,1 ./build-sm70/strata-device --devices 0,1 --p2p --selftest`,
record the bidirectional GB/s into `Docs/v100-release2-multigpu.md`.

## 6. Step 3.3 — design notes for the direct P2P hand-off

Facts established by the survey that make this a small change:

- Ordering is host-guaranteed today: each stage syncs its own stream (verify.cpp L1336) before
  calling `next_->run` (L1383–1385). A direct P2P write either reuses that ordering or adds
  CUDA events on the stage streams; the existing sync is the safe fallback.
- The copy kernel needs no change (`copy_from_mapped` dereferences whatever device pointer it
  is given, elementwise.cu L226).
- Plan check: `devices.p2p[stage_dev][next_dev] == 1` (DevicePlan semantics above).
- Per the task: `cudaMemcpyPeerAsync` + CUDA events, keep pinned host staging fallback,
  startup log exactly:
  `strata serve: layer split hand-off: direct P2P (cudaMemcpyPeerAsync)` (plus a fallback
  line for the non-P2P path).
- The hand-off buffers (generate.cpp L4316–4327) are per-request-window allocations sized
  `kVerifyMaxT * handoff_floats * 4` bytes — small; the P2P variant allocates the same buffer
  on the *next* stage's device so the earlier stage's GPU writes it peer-to-peer.
- `src/core/layer.cpp` L382 (doorbell D2D) is the second place to inspect if the stage-1
  path touches hand-off state beyond the verifier.

## 7. Pending checklist (resume here)

1. **3.1** `cd ~/dsh/strata/Strata`; `git checkout main` (create local tracking
   `origin/main` if needed); `git merge release3-qwen35-dense -m "Merge 0.1.35 sync from
   release3-qwen35-dense into main"`; `git push origin main`;
   `git checkout -b release2-v100-multigpu`; `git push -u origin release2-v100-multigpu`.
   (Untracked Logs/bench artifacts don't block; handle the modified `Logs/benchmarks/*`
   files as needed for a clean merge.)
2. **3.2** apply the device.cu fix (§5), rebuild, run the probe, record GB/s.
3. **3.3** implement the direct P2P hand-off (§6) + fallback + startup log.
4. **3.4** verify residency: 12,288 experts/card in cache logs
   (generate.cpp L3024 prints per-stage cache slots/GiB), `--ple-io ram` default,
   int8 KV filling the remaining ~10–12 GiB/card.
5. **3.5** clean SM70 build + ctest (expect 30/31).
6. **3.6** 2-GPU correctness run (`CUDA_VISIBLE_DEVICES=0,1 --layer-split auto`,
   `--serve`), verify first token 271 + golden 32-prefix + balanced VRAM + 0 CUDA errors;
   then `s110.sh ctx 16384` / `ctx 32768` benchmarks vs Strata-Pure
   (`/home/noorazman/Strata-Pure`), capture TTFT, prefill/decode tok/s, spec acceptance,
   VRAM peaks, P2P GB/s.
7. **3.7** `Docs/v100-release2-multigpu.md`, commit, push,
   `~/.dsh/bin/dsh-notify "Release 2: 2x V100 32GB multi-GPU P2P setup and benchmarks
   complete"`, final structured summary vs Strata-Pure and Release 1.

## 8. Conventions + constraints

- One concise `~/.dsh/bin/dsh-notify` per significant event; final message text is fixed
  (see step 3.7). Never treat a notify failure as task failure.
- Stop `strata.service` before gates (it is inactive now; GPU 0 shows 4 MiB).
- Do NOT request sandbox escalation; file policy is danger-full-access.
- Do NOT use `Docs/v100-release2-phase2-resume.md` as a reference (user: "put that aside").
- Hindsight: "V100 Release 2 — Multi-GPU" (kp-7c4309d7e07f480caf3d2e7a17eb8c36) and
  "V100 Release 1 — production freeze & validation" (kp-3f39ff3b77084d349f6863d4ba0d3eec)
  pages exist but were regenerating at pause time — search again if useful.
- Workspace root is NOT the git repo root; the repo is `~/dsh/strata/Strata`.
- `--layer-split` only exists in `--serve` mode.
