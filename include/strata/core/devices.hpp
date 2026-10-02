// include/strata/core/devices.hpp - Release 2, phase 1: the multi-device plan.
//
// The device plan for one engine run is built BEFORE the first allocation, for the same reason the memory
// plan is printed against `cudaMemGetInfo` (device.hpp): a run that discovers at token 4000 that the second
// card is not what it expected has already lost.  Phase 1 is the foundation only - the plan is built, the
// peer connections are established and reported, and the default single-device path stays bit-identical.
// Phase 2 places the context-scaling state (the QSA KV cache, the block-pooled indexer state, the RoPE
// tables) on the auxiliary device and re-captures the layer graphs.
//
// The plan is device-agnostic by construction: nothing here may assume the cards are the same size, the
// same NUMA node, or connected by anything faster than PCIe.  Development runs on 0,2 (a 32 GB PCIe card
// plus a 16 GB SXM2, different NUMA nodes); the final target is 0,1 (two 32 GB cards).
#pragma once

#include "strata/core/device.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace strata::core {

struct DevicePlan {
    std::vector<int> ordinals;
    std::vector<DeviceInfo> info;              // parallel to ordinals
    // p2p[i][j] == 1 when device `ordinals[j]`'s memory is DIRECTLY addressable FROM device `ordinals[i]`:
    // peer access was not merely reported possible by cudaDeviceCanAccessPeer but actually ENABLED at plan
    // time (the diagonal is 1 by definition).  A zero where phase 2 will need a transfer is the case the
    // caller must stage through the host - that is the fallback, and the report says so at startup.
    std::vector<std::vector<int>> p2p;
    // p2p_gbps[i][j] > 0: the measured ordinals[i] <- ordinals[j] device-to-device bandwidth in GB/s, filled
    // only when the plan was built with probe_bandwidth (the strata-device self-test).  The engine path
    // leaves it zero so a boot stays fast; the number a placement decision needs is measured, not assumed.
    std::vector<std::vector<double>> p2p_gbps;

    int primary() const { return ordinals.empty() ? -1 : ordinals.front(); }
    bool multi() const { return ordinals.size() > 1; }
    int index_of(int ordinal) const;
};

/// Parses a device spec ("0", "0,2", "0,1,3") and builds the plan: per-device `device_info` (which also
/// enforces the sm_70 floor on every card, not just the primary), then pairwise
/// `cudaDeviceCanAccessPeer` + `cudaDeviceEnablePeerAccess` on every usable pair.  Throws `CudaError` on a
/// bad or duplicate ordinal or an enable the driver refuses.  `probe_bandwidth` additionally measures the
/// real D2D bandwidth of every ENABLED pair (timed `cudaMemcpyPeer`) - the self-test sets it, the engine
/// does not.
DevicePlan make_device_plan(const std::string& spec, bool probe_bandwidth = false);

/// The startup report, one line, e.g.
///   strata devices: primary 0 (Tesla V100-PCIE-32GB, sm_70, 31.75 GiB free / 31.99 GiB total);
///   aux 2 (Tesla V100-SXM2-16GB, sm_70, 15.63 GiB free / 15.99 GiB total); p2p 0->2, 2->0
/// The p2p edges are DIRECTIONAL: "0->2" means memory of device 2 is directly addressable from device 0
/// (0 reads 2); both directions listed = bidirectional.
std::string device_plan_report(const DevicePlan& plan);

}  // namespace strata::core
