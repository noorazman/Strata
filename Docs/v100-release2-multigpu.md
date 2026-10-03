# Release 2: 2× V100 32GB multi-GPU (P2P layer split)

Machine: 5-card box, cards 0 and 1 = V100-PCIE-32GB on the same PCIe root complex (no
NVLink between them), both on NUMA node 0. Cards 2 and 3 are V100-SXM2-16GB; card 4 runs a
separate single-GPU llama-server tenant.

## P2P probe (step 3.2)

`strata-device --devices 0,1 --p2p` enables peer access between the two cards
(`cudaDeviceCanAccessPeer` + `cudaDeviceEnablePeerAccess`) *before* measuring, so the number
is a true device-to-device transfer, not a host-staged fallback. Measurement is a
`cudaMemcpyPeer` loop bracketed by CUDA events, with a warmup pass first.

Measured on 2026-10-03, cards clean (4 MiB each, no other processes):

| direction | GB/s |
|---|---|
| 0 → 1 | 1.31 |
| 1 → 0 | 1.52 |

Post-reboot retest (2026-10-03, cards clean, `Recovery Action: None`, 0 Xids since boot):
0 → 1 = 1.59 GB/s, 1 → 0 = 1.30 GB/s, and the probe now **exits cleanly** (exit 0). The
per-direction numbers sit in the same 1.3–1.6 GB/s band on both runs; which direction is
faster flips run to run, so treat the pair as ~1.3–1.6 GB/s each way. The clean exit also
settles the earlier question: the original post-`main` exit hang was residual state from
the stuck probe PIDs (their Xid 31 MMU faults), not a driver defect that needed a reboot.

The asymmetric directions are expected: root-complex PCIe P2P does not route both ways
equally, and the two cards sit on different downstream ports of the root complex.

Two probe bugs were found and fixed during 3.2:

- `measure_p2p_gbps` seeded its source buffer with `cudaMemset` *after* switching the active
  device to the destination, so the memset landed on the wrong card (`invalid argument`).
  The seed now runs on the source device, before `cudaSetDevice(dst)`.
- `device_plan_report` printed the GB/s value × 100 with an "MB/s" label, so a real
  1.31 GB/s link read "131 MB/s". It now multiplies by 1000.

## Driver wedge and recovery

After the probe runs, the cards stayed in use until the stuck probe PIDs were killed.
At 10:56 on 2026-10-03 the kernel logged a storm: CMCI storm on CPU0 BANK5, machine-check
events, and Xid 31 (MMU fault, `FAULT_PTE`, pid=modprobe) on all five cards; the driver
then set Xid 154 "GPU recovery action changed to 0x2 (Node Reboot Required)" on every
card. After that, `cudaGetDeviceCount` failed on all cards, not just 0 and 1.

`nvidia-smi --gpu-reset` was not possible without root: the `nvidia-persistenced` daemon
re-enables persistence mode within ~100 ms of `-pm 0`, and `systemctl stop` needs polkit
interactive auth. Recovery is a node reboot; after the reboot cards 0 and 1 are re-tested
with the same `strata-device` command to confirm the numbers and a clean exit.

The same wedge recurred a second time, and this time the trigger was unambiguously the probe
itself. The post-reboot retest ran at ~13:37:30–40 (between a clean `journalctl -k` check at
13:37:21 and the storm), on exactly the two cards it faults. The kernel then logged at
13:37:50 a DMAR `PTE Write access is not set` fault on 03:00.0 (GPU 1) and at 13:37:51 an
Xid 31 MMU fault (`FAULT_PTE`, `ACCESS_TYPE_VIRT_WRITE`) on 02:00.0 (GPU 0) — the NVRM line
attributes it to `pid=1456, name=modprobe`, a logging quirk: faults raised on kernel
DMA/teardown paths are recorded under the module name — and Xid 154 "Node Reboot Required"
on all five cards again. The separate llama-server tenant on card 4 was not touched by the
probe, but with the driver wedged its `cuInit(0)` failed with error 999 (`/dev/nvidia-uvm`
EIO) and the server fell back to CPU. The second reboot (14:14:16) cleared all five cards
(`GPU Recovery Action: None`, 0 Xids since the boot) and llama-server came back on GPU 4
(PID 1360, ~15.5 GB, port 8081).

Both storms followed a `strata-device` P2P probe run within ~30 s, and a non-fatal
`mce: Machine check events logged` (same CMCI/BANK5 family as the 10:56 storm) appeared at
14:18:20, minutes after the second boot with no probe running. Read together this is a
hardware-level fault on the shared root complex that P2P probe traffic merely triggers. The
probe traffic is therefore kept short (128 MiB blocks, 250 ms / 1 GiB cap) and `journalctl -k`
is watched during the 2-GPU serve runs (3.3/3.6); if storms recur, the P2P traffic is limited
or the runs are moved to a quiet window.

## Hand-off wiring (step 3.3)

(To be completed: direct P2P stage hand-off in `src/program/generate.cpp` / `src/core/layer.cpp`,
startup log, pinned-host fallback.)

## Correctness and benchmarks (steps 3.4–3.6)

(To be completed: 100% expert residency, PLE in RAM, int8 KV; 2-GPU correctness run; 16K/32K
benchmarks vs Strata-Pure.)
