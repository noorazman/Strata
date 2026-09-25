# TASKS

## NOW
- (idle) Stage 1 complete; PR closed; build local. No active work.

## NEXT
- On user request: reopen/replace the upstream PR (branch `feature/v100-moe` is on `origin`), or start Stage 2.
- Optional housekeeping: make `noorazman/Strata` private later (Settings → Unfork → change visibility; API unfork needs admin token scope).

## LATER
- Stage 2 candidates (only after Stage 1 merges / user direction): dense-model path, multi-GPU, additional quants, scheduler/memory rewrites, new API frameworks.
- If a 16 GB deploy is wanted long-term: consider a smaller explicit `--expert-cache` or int8 KV at 8192+ ctx to grow the VRAM headroom.

## FUTURE
- Re-verify when the upstream hit path is fixed (ROUND 328) — the token-match comparison in `Docs/v100-testing.md` is the regression check.
- Q2_0 fixtures for the `ple_parity` ctest (currently red by design).
