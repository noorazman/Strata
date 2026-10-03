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

## Hand-off wiring (step 3.3)

(To be completed: direct P2P stage hand-off in `src/program/generate.cpp` / `src/core/layer.cpp`,
startup log, pinned-host fallback.)

## Correctness and benchmarks (steps 3.4–3.6)

(To be completed: 100% expert residency, PLE in RAM, int8 KV; 2-GPU correctness run; 16K/32K
benchmarks vs Strata-Pure.)
