// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "iree/compiler/Dialect/Flow/Transforms/Passes.h"
#include "iree/compiler/Dialect/HAL/IR/HALDialect.h"
#include "iree/compiler/Dialect/HAL/IR/HALTypes.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

namespace mlir::iree_compiler::IREE::Flow {

#define GEN_PASS_DEF_ASSIGNDEVICEAFFINITYPASS
#include "iree/compiler/Dialect/Flow/Transforms/Passes.h.inc"

namespace {

struct DeviceRange {
  unsigned deviceId;
  unsigned start; // inclusive
  unsigned end;   // inclusive
};

// Parses a device map string of the form "device_id:start-end,..." into a
// vector of DeviceRange structs. Returns failure on malformed input.
static LogicalResult parseDeviceMap(StringRef mapStr,
                                    SmallVectorImpl<DeviceRange> &ranges) {
  if (mapStr.empty())
    return failure();

  SmallVector<StringRef> entries;
  mapStr.split(entries, ',');

  for (auto entry : entries) {
    // Each entry is "device_id:start-end"
    auto [deviceStr, rangeStr] = entry.split(':');
    if (deviceStr.empty() || rangeStr.empty())
      return failure();

    auto [startStr, endStr] = rangeStr.split('-');
    if (startStr.empty() || endStr.empty())
      return failure();

    unsigned deviceId, start, end;
    if (deviceStr.getAsInteger(10, deviceId) ||
        startStr.getAsInteger(10, start) || endStr.getAsInteger(10, end))
      return failure();

    if (start > end)
      return failure();

    ranges.push_back({deviceId, start, end});
  }

  return success();
}

struct AssignDeviceAffinityPass
    : public IREE::Flow::impl::AssignDeviceAffinityPassBase<
          AssignDeviceAffinityPass> {
  using IREE::Flow::impl::AssignDeviceAffinityPassBase<
      AssignDeviceAffinityPass>::AssignDeviceAffinityPassBase;

  void runOnOperation() override {
    auto moduleOp = getOperation();
    auto *ctx = moduleOp.getContext();

    if (deviceMap.empty()) {
      moduleOp.emitError("device-map option is required");
      return signalPassFailure();
    }

    // Parse device map.
    SmallVector<DeviceRange> ranges;
    if (failed(parseDeviceMap(deviceMap, ranges))) {
      moduleOp.emitError("malformed device-map: expected format "
                         "'device_id:start-end[,device_id:start-end,...]'");
      return signalPassFailure();
    }

    // Sort by start index.
    llvm::sort(ranges,
               [](const DeviceRange &a, const DeviceRange &b) {
                 return a.start < b.start;
               });

    // Validate: first range must start at 0.
    if (ranges[0].start != 0) {
      moduleOp.emitError("device-map: first range must start at 0, got ")
          << ranges[0].start;
      return signalPassFailure();
    }

    // Validate: no gaps and no overlaps between consecutive ranges.
    for (unsigned i = 1; i < ranges.size(); ++i) {
      if (ranges[i].start != ranges[i - 1].end + 1) {
        moduleOp.emitError("device-map: ranges must be contiguous with no "
                           "gaps or overlaps; range starting at ")
            << ranges[i].start << " does not follow range ending at "
            << ranges[i - 1].end;
        return signalPassFailure();
      }
    }

    // Build affinity attribute for each unique device ID.
    DenseMap<unsigned, IREE::HAL::DeviceAffinityAttr> affinityMap;
    for (auto &r : ranges) {
      if (!affinityMap.count(r.deviceId)) {
        std::string deviceName = "__device_" + std::to_string(r.deviceId);
        auto ref = FlatSymbolRefAttr::get(ctx, deviceName);
        affinityMap[r.deviceId] =
            IREE::HAL::DeviceAffinityAttr::get(ctx, ref, /*queue_mask=*/-1);
      }
    }

    auto affinityAttrName = StringAttr::get(ctx, "stream.affinity");

    // Walk all flow.dispatch ops in program order and assign affinities.
    unsigned dispatchIndex = 0;
    moduleOp->walk([&](IREE::Flow::DispatchOp dispatchOp) {
      // Find the range containing this dispatch index.
      for (auto &r : ranges) {
        if (dispatchIndex >= r.start && dispatchIndex <= r.end) {
          dispatchOp->setAttr(affinityAttrName, affinityMap[r.deviceId]);
          break;
        }
      }
      ++dispatchIndex;
    });

    // Verify that all dispatches were covered.
    unsigned expectedMax = ranges.back().end;
    if (dispatchIndex > 0 && expectedMax != dispatchIndex - 1) {
      moduleOp.emitWarning("device-map covers dispatches 0-")
          << expectedMax << " but module has " << dispatchIndex
          << " dispatches";
    }
  }
};

} // namespace

} // namespace mlir::iree_compiler::IREE::Flow
