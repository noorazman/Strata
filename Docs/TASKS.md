# TASKS

## NOW
- (idle) V100 Stage 1.1 complete — `--pool-workers 24` adopted (+1.0 % decode, 32/32 golden); 16 GB GPU4 regression PASS. All docs + raw data committed on `stage1.1-performance`.

## NEXT
- On user request: push `stage1.1-performance` to `origin`, start Stage 2, or take on an engine-level decode item (below).
- Engine-level decode levers (from `Docs/v100-stage1.1-final.md`): verify-window cost (95 % of GPU busy), `wait_flag_ge`→CUDA events (37 % of GPU busy is spin), pool dequant bandwidth (22.3 GB/s floor).
- Regenerate a larger expert profile (`tools/make_profile.py` referenced in-source but missing from tree; 8,000-pair cap holds hits at 87.6 %).
- Report the `--expert-cache-per-layer` + profile-fill `verify_slot` mismatch bug upstream.
- Optional housekeeping: make `noorazman/Strata` private later (Settings → Unfork → change visibility; API unfork needs admin token scope).

## LATER
- Stage 2 candidates (only after user direction): dense-model path, multi-GPU, additional quants, scheduler/memory rewrites, new API frameworks.
- If a 16 GB deploy is wanted long-term: fits today (16.13 GiB peak, workers 24); optional smaller explicit `--expert-cache` or int8 KV for more headroom.

## FUTURE
- Re-verify when the upstream hit path is fixed (ROUND 328) — the token-match comparison in `Docs/v100-testing.md` is the regression check.
- Q2_0 fixtures for the `ple_parity` ctest (currently red by design).
