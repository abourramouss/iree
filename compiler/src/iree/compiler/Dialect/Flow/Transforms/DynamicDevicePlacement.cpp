// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "iree/compiler/Dialect/Flow/Transforms/Passes.h"
#include "iree/compiler/Dialect/HAL/IR/HALDialect.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/Dialect/HAL/IR/HALTypes.h"
#include "iree/compiler/Dialect/Util/IR/UtilOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"

namespace mlir::iree_compiler::IREE::Flow {

#define GEN_PASS_DEF_DYNAMICDEVICEPLACEMENTPASS
#include "iree/compiler/Dialect/Flow/Transforms/Passes.h.inc"

namespace {

struct DynamicDevicePlacementPass
    : public IREE::Flow::impl::DynamicDevicePlacementPassBase<
          DynamicDevicePlacementPass> {
  using IREE::Flow::impl::DynamicDevicePlacementPassBase<
      DynamicDevicePlacementPass>::DynamicDevicePlacementPassBase;

  void runOnOperation() override {
    auto moduleOp = getOperation();
    auto *ctx = moduleOp.getContext();

    // 1. Collect all flow.dispatch ops in program order.
    SmallVector<IREE::Flow::DispatchOp> dispatches;
    moduleOp->walk([&](IREE::Flow::DispatchOp op) {
      dispatches.push_back(op);
    });
    if (dispatches.empty())
      return;

    int64_t numDispatches = static_cast<int64_t>(dispatches.size());

    // 2. Find the containing util.func (the async entry function).
    auto funcOp = dispatches[0]->getParentOfType<IREE::Util::FuncOp>();
    if (!funcOp) {
      moduleOp.emitError("flow.dispatch ops not inside a util.func");
      return signalPassFailure();
    }

    auto loc = funcOp.getLoc();
    auto bufferViewType = IREE::HAL::BufferViewType::get(ctx);

    // 3. Add placement !hal.buffer_view argument to the function.
    //    Insert at position 1 (after the first buffer_view, before fences).
    unsigned insertPos = 1;
    auto &entryBlock = funcOp.getBody().front();
    auto placementArg =
        entryBlock.insertArgument(insertPos, bufferViewType, loc);

    // Update function type.
    auto funcType = funcOp.getFunctionType();
    SmallVector<Type> inputTypes(funcType.getInputs());
    inputTypes.insert(inputTypes.begin() + insertPos, bufferViewType);
    auto newFuncType =
        FunctionType::get(ctx, inputTypes, funcType.getResults());
    funcOp.setFunctionType(newFuncType);

    // 4. Create hal.tensor.import for the placement tensor.
    //    Find existing hal.tensor.import to get the wait fence value.
    IREE::HAL::TensorImportOp existingImport;
    funcOp->walk([&](IREE::HAL::TensorImportOp importOp) {
      if (!existingImport)
        existingImport = importOp;
    });

    if (!existingImport) {
      moduleOp.emitError("no hal.tensor.import found in function");
      return signalPassFailure();
    }

    Value waitFence = existingImport.getWaitFence();

    auto i32Type = IntegerType::get(ctx, 32);
    auto placementTensorType =
        RankedTensorType::get({numDispatches}, i32Type);

    OpBuilder importBuilder(ctx);
    importBuilder.setInsertionPointAfter(existingImport);

    auto placementImport = IREE::HAL::TensorImportOp::create(
        importBuilder, loc, placementTensorType, placementArg,
        TypeAttr::get(placementTensorType),
        /*consume=*/false, waitFence,
        /*name=*/StringAttr{}, /*affinity=*/Attribute{});

    Value placementTensor = placementImport.getResult();

    // 5. Create comparison constant (i32 zero).
    Value zeroI32 = arith::ConstantOp::create(
        importBuilder, loc, importBuilder.getI32IntegerAttr(0));

    // 6. Build affinity attributes for device 0 and device 1.
    auto dev0Ref = FlatSymbolRefAttr::get(ctx, "__device_0");
    auto dev0 =
        IREE::HAL::DeviceAffinityAttr::get(ctx, dev0Ref, /*queue_mask=*/-1);
    auto dev1Ref = FlatSymbolRefAttr::get(ctx, "__device_1");
    auto dev1 =
        IREE::HAL::DeviceAffinityAttr::get(ctx, dev1Ref, /*queue_mask=*/-1);

    auto affinityAttrName = StringAttr::get(ctx, "stream.affinity");

    // 7. Wrap each dispatch in scf.if/else gated by placement[i].
    for (int64_t i = 0; i < numDispatches; ++i) {
      auto dispatchOp = dispatches[i];
      OpBuilder builder(dispatchOp);

      // Extract placement[i] and compare with zero.
      Value idx = arith::ConstantIndexOp::create(builder, loc, i);
      Value pval = tensor::ExtractOp::create(
          builder, loc, placementTensor, ValueRange{idx});
      Value cond = arith::CmpIOp::create(
          builder, loc, arith::CmpIPredicate::eq, pval, zeroI32);

      // Build scf.if: then branch → device 0, else branch → device 1.
      auto ifOp = scf::IfOp::create(
          builder, loc, dispatchOp->getResultTypes(), cond,
          /*addThenBlock=*/true, /*addElseBlock=*/true);

      // Then branch: dispatch on device 0.
      {
        auto thenBuilder = OpBuilder::atBlockBegin(ifOp.thenBlock());
        auto *cloned = thenBuilder.clone(*dispatchOp);
        cloned->setAttr(affinityAttrName, dev0);
        scf::YieldOp::create(thenBuilder, loc, cloned->getResults());
      }

      // Else branch: dispatch on device 1.
      {
        auto elseBuilder = OpBuilder::atBlockBegin(ifOp.elseBlock());
        auto *cloned = elseBuilder.clone(*dispatchOp);
        cloned->setAttr(affinityAttrName, dev1);
        scf::YieldOp::create(elseBuilder, loc, cloned->getResults());
      }

      dispatchOp->replaceAllUsesWith(ifOp.getResults());
      dispatchOp->erase();
    }

    // 8. Update callers: walk for util.call ops referencing the modified
    //    function and propagate the new placement argument.
    SmallVector<IREE::Util::CallOp> callsToUpdate;
    moduleOp->walk([&](IREE::Util::CallOp callOp) {
      if (callOp.getCallee() == funcOp.getName())
        callsToUpdate.push_back(callOp);
    });

    for (auto callOp : callsToUpdate) {
      auto callerFunc = callOp->getParentOfType<IREE::Util::FuncOp>();
      if (!callerFunc)
        continue;

      // Add placement argument to the caller's signature.
      auto &callerEntry = callerFunc.getBody().front();
      auto newCallerArg =
          callerEntry.insertArgument(insertPos, bufferViewType, loc);

      auto callerFuncType = callerFunc.getFunctionType();
      SmallVector<Type> callerInputTypes(callerFuncType.getInputs());
      callerInputTypes.insert(callerInputTypes.begin() + insertPos,
                              bufferViewType);
      auto newCallerFuncType =
          FunctionType::get(ctx, callerInputTypes,
                            callerFuncType.getResults());
      callerFunc.setFunctionType(newCallerFuncType);

      // Rebuild the call with the new operand inserted at insertPos.
      SmallVector<Value> newOperands;
      auto oldOperands = callOp.getArgOperands();
      for (unsigned j = 0; j < oldOperands.size(); ++j) {
        if (j == insertPos)
          newOperands.push_back(newCallerArg);
        newOperands.push_back(oldOperands[j]);
      }
      if (insertPos >= oldOperands.size())
        newOperands.push_back(newCallerArg);

      OpBuilder callBuilder(callOp);
      auto newCall = IREE::Util::CallOp::create(
          callBuilder, callOp.getLoc(),
          callOp.getResultTypes(),
          callOp.getCalleeAttr(),
          newOperands,
          /*tied_operands=*/ArrayAttr{},
          /*arg_attrs=*/ArrayAttr{},
          /*res_attrs=*/ArrayAttr{});
      callOp->replaceAllUsesWith(newCall->getResults());
      callOp->erase();
    }
  }
};

} // namespace

} // namespace mlir::iree_compiler::IREE::Flow
