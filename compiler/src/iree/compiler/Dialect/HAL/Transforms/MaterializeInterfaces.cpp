// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <memory>
#include <utility>

#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "iree/compiler/Dialect/HAL/Analysis/BindingLayout.h"
#include "iree/compiler/Dialect/HAL/IR/HALDialect.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/Dialect/HAL/Target/TargetBackend.h"
#include "iree/compiler/Dialect/HAL/Target/TargetRegistry.h"
#include "iree/compiler/Dialect/HAL/Transforms/Passes.h"
#include "iree/compiler/Dialect/Stream/IR/StreamOps.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-hal-materialize-interfaces"

namespace mlir::iree_compiler::IREE::HAL {

#define GEN_PASS_DEF_MATERIALIZEINTERFACESPASS
#include "iree/compiler/Dialect/HAL/Transforms/Passes.h.inc"

namespace {

// Map of original SymbolRefAttr to a list of SymbolRefAttrs in variants marked
// with the executable target the export is assigned.
using ExportExpansions = DenseMap<
    Attribute,
    SmallVector<std::pair<Attribute, IREE::HAL::ExecutableTargetAttr>>>;

// Map of operations (executables, dispatches, etc) to the executable targets
// required by those operations based on usage. If missing or empty the default
// set should be used.
using RequiredExecutableTargets =
    DenseMap<Operation *, SetVector<IREE::HAL::ExecutableTargetAttr>>;

//===----------------------------------------------------------------------===//
// Utilities
//===----------------------------------------------------------------------===//

static SymbolRefAttr
makeExportSymbolRefAttr(IREE::HAL::ExecutableOp executableOp,
                        IREE::HAL::ExecutableVariantOp variantOp,
                        IREE::HAL::ExecutableExportOp exportOp) {
  return SymbolRefAttr::get(executableOp.getNameAttr(),
                            {
                                FlatSymbolRefAttr::get(variantOp.getNameAttr()),
                                FlatSymbolRefAttr::get(exportOp.getNameAttr()),
                            });
}

static void setApplicableObjects(Operation *sourceOp,
                                 IREE::HAL::ExecutableVariantOp targetOp) {
  auto objectsAttr = sourceOp->getAttrOfType<IREE::HAL::ExecutableObjectsAttr>(
      "hal.executable.objects");
  if (!objectsAttr)
    return;
  auto objects = objectsAttr.getApplicableObjects(targetOp.getTarget());
  if (!objects)
    return;
  targetOp.setObjectsAttr(*objects);
}

// Returns a set of target devices that may be active for the given
// operation. This will recursively walk parent operations until one with
// the `hal.device.targets` attribute is found.
//
// This is a legacy mechanism for performing the search. Newer code should use
// affinities or !hal.device analysis instead.
static void gatherLegacyDeviceTargetAttrs(
    Operation *op, SetVector<IREE::HAL::DeviceTargetAttr> &resultSet) {
  auto attrId = StringAttr::get(op->getContext(), "hal.device.targets");
  while (op) {
    auto targetsAttr = op->getAttrOfType<ArrayAttr>(attrId);
    if (targetsAttr) {
      for (auto targetAttr : targetsAttr)
        resultSet.insert(llvm::cast<IREE::HAL::DeviceTargetAttr>(targetAttr));
      return;
    }
    op = op->getParentOp();
  }
  // No devices found; let caller decide what to do.
}

// Returns a set of target executable formats that may be active for the given
// operation.
//
// This is a legacy mechanism for performing the search. Newer code should use
// affinities or !hal.device analysis instead.
static void gatherLegacyExecutableTargetAttrs(
    Operation *op, SetVector<IREE::HAL::ExecutableTargetAttr> &resultSet) {
  SetVector<IREE::HAL::DeviceTargetAttr> deviceTargetSet;
  gatherLegacyDeviceTargetAttrs(op, deviceTargetSet);
  for (auto deviceTargetAttr : deviceTargetSet) {
    deviceTargetAttr.getExecutableTargets(resultSet);
  }
}

// Gathers the set of device targets potentially referenced by the given
// affinity. Targets are ordered by most likely to least likely.
// Returns failure if analysis fails and the result set is incomplete.
static void
gatherDeviceAffinityTargets(IREE::Stream::AffinityAttr affinityAttr,
                            Operation *fromOp, SymbolTable &symbolTable,
                            SetVector<IREE::HAL::DeviceTargetAttr> &resultSet) {
  // We currently only know how to handle HAL device affinities.
  // We could support other ones via an interface but instead we just fall back
  // to default logic if no affinity or an unknown one is found.
  auto deviceAffinityAttr =
      dyn_cast_if_present<IREE::HAL::DeviceAffinityAttr>(affinityAttr);
  if (!deviceAffinityAttr) {
    gatherLegacyDeviceTargetAttrs(fromOp, resultSet);
    return;
  }

  // Recursively resolve the referenced device into targets.
  SetVector<Attribute> worklist;
  worklist.insert(deviceAffinityAttr.getDevice());
  do {
    auto attr = worklist.pop_back_val();
    if (!TypeSwitch<Attribute, bool>(attr)
             .Case<SymbolRefAttr>([&](auto symRefAttr) {
               auto globalOp =
                   symbolTable
                       .lookupNearestSymbolFrom<IREE::Util::GlobalOpInterface>(
                           fromOp, symRefAttr);
               assert(globalOp && "global reference must be valid");
               if (auto initialValueAttr = globalOp.getGlobalInitialValue()) {
                 // Global with a device initialization value we can analyze.
                 worklist.insert(initialValueAttr);
                 return true;
               } else {
                 return false;
               }
             })
             .Case<IREE::HAL::DeviceTargetAttr>([&](auto targetAttr) {
               resultSet.insert(targetAttr);
               return true;
             })
             .Case<IREE::HAL::DeviceFallbackAttr>([&](auto fallbackAttr) {
               worklist.insert(fallbackAttr.getName());
               return true;
             })
             .Case<IREE::HAL::DeviceSelectAttr>([&](auto selectAttr) {
               worklist.insert(selectAttr.getDevices().begin(),
                               selectAttr.getDevices().end());
               return true;
             })
             .Default([](auto attr) { return false; })) {
      // No initial value means fall back to defaults. We do that by
      // inserting all knowable targets.
      gatherLegacyDeviceTargetAttrs(fromOp, resultSet);
      return;
    }
  } while (!worklist.empty());
}

// Gathers all executable targets that may be used by the given dispatch op.
static void gatherRequiredExecutableTargets(
    IREE::Stream::CmdDispatchOp dispatchOp, SymbolTable &symbolTable,
    SetVector<IREE::HAL::ExecutableTargetAttr> &resultSet) {
  // Get the affinity from the op or an ancestor. Note that there may be no
  // affinity specified at all.
  auto affinityAttr = IREE::Stream::AffinityAttr::lookup(dispatchOp);

  // Gather the device targets that are referenced by the affinity.
  SetVector<IREE::HAL::DeviceTargetAttr> deviceTargetSet;
  gatherDeviceAffinityTargets(affinityAttr, dispatchOp, symbolTable,
                              deviceTargetSet);

  // Add all executable targets on the device targets.
  for (auto deviceTargetAttr : deviceTargetSet) {
    resultSet.insert(deviceTargetAttr.getExecutableTargets().begin(),
                     deviceTargetAttr.getExecutableTargets().end());
  }
}

// Builds a map of executable and dispatch ops to the executable targets that
// may be required.
static RequiredExecutableTargets
buildRequiredExecutableTargetsMap(ModuleOp moduleOp, SymbolTable &symbolTable,
                                  BindingLayoutAnalysis &layoutAnalysis) {
  RequiredExecutableTargets resultMap;
  for (auto executableOp : moduleOp.getOps<IREE::HAL::ExecutableSourceOp>()) {
    auto &executableTargetAttrs = resultMap[executableOp];
    for (auto exportOp : executableOp.getOps<IREE::HAL::ExecutableExportOp>()) {
      for (auto dispatchOp : layoutAnalysis.getExportDispatches(exportOp)) {
        auto &dispatchTargetAttrs = resultMap[dispatchOp];
        gatherRequiredExecutableTargets(dispatchOp, symbolTable,
                                        dispatchTargetAttrs);
        executableTargetAttrs.insert(dispatchTargetAttrs.begin(),
                                     dispatchTargetAttrs.end());
      }
    }
    if (executableOp.isPublic()) {
      // Public executables need all possible targets.
      gatherLegacyExecutableTargetAttrs(executableOp, executableTargetAttrs);
    }
  }
  for (auto executableOp : moduleOp.getOps<IREE::Stream::ExecutableOp>()) {
    auto &executableTargetAttrs = resultMap[executableOp];
    for (auto exportOp :
         executableOp.getOps<IREE::Stream::ExecutableExportOp>()) {
      for (auto dispatchOp : layoutAnalysis.getExportDispatches(exportOp)) {
        auto &dispatchTargetAttrs = resultMap[dispatchOp];
        gatherRequiredExecutableTargets(dispatchOp, symbolTable,
                                        dispatchTargetAttrs);
        executableTargetAttrs.insert(dispatchTargetAttrs.begin(),
                                     dispatchTargetAttrs.end());
      }
    }
    if (executableOp.isPublic()) {
      // Public executables need all possible targets.
      gatherLegacyExecutableTargetAttrs(executableOp, executableTargetAttrs);
    }
  }
  return resultMap;
}

// Updates the target entry point symbols of |dispatchOp| to the expanded set of
// variant exports in |exportExpansions|.
static void
updateDispatchTargets(IREE::Stream::CmdDispatchOp dispatchOp,
                      const ExportExpansions &exportExpansions,
                      RequiredExecutableTargets &requiredExecutableTargets) {
  auto &requiredTargetAttrs = requiredExecutableTargets[dispatchOp];
  SmallVector<Attribute> newAttrs;
  for (auto oldAttr : dispatchOp.getEntryPointRefs()) {
    auto it = exportExpansions.find(oldAttr);
    if (it == exportExpansions.end()) {
      newAttrs.push_back(oldAttr); // preserve existing
      continue;
    }
    for (auto [newAttr, targetAttr] : it->second) {
      // Filter the new expansions to only those used by the dispatch (if we
      // have a valid filter).
      if (requiredTargetAttrs.empty())
        newAttrs.push_back(newAttr);
      else if (requiredTargetAttrs.contains(targetAttr))
        newAttrs.push_back(newAttr);
    }
  }
  dispatchOp.setEntryPointsAttr(
      ArrayAttr::get(dispatchOp.getContext(), newAttrs));
}

//===----------------------------------------------------------------------===//
// hal.executable.source materialization
//===----------------------------------------------------------------------===//

static void materializeExecutableFromSourceOp(
    IREE::HAL::ExecutableSourceOp sourceOp,
    BindingLayoutAnalysis &layoutAnalysis,
    RequiredExecutableTargets &requiredExecutableTargets) {
  // Gather the required executable targets based on the dispatches to exports
  // in the source op.
  SmallVector<IREE::HAL::ExecutableTargetAttr> targetAttrs(
      requiredExecutableTargets[sourceOp].getArrayRef());
  if (targetAttrs.empty())
    return;
  llvm::stable_sort(targetAttrs, [](auto lhs, auto rhs) {
    return lhs.getSymbolNameFragment() < rhs.getSymbolNameFragment();
  });

  // Create the op that will contain the translated executable.
  OpBuilder moduleBuilder(sourceOp);
  auto executableOp = moduleBuilder.create<IREE::HAL::ExecutableOp>(
      sourceOp.getLoc(), sourceOp.getName());
  executableOp.setVisibility(sourceOp.getVisibility());

  // With this hand-authored path all variants have the same layout and entry
  // points and we can just clone them.
  auto sourceExportOps = sourceOp.getExportOps();

  // Materialize all of the hal.executable.variant ops for all backends we are
  // targeting.
  ExportExpansions exportExpansions;
  SymbolTable targetSymbolTable(executableOp);
  OpBuilder targetBuilder(&executableOp.getBlock().back());
  for (auto targetAttr : targetAttrs) {
    // Create new variant and clone the exports.
    auto targetVariantOp = targetBuilder.create<IREE::HAL::ExecutableVariantOp>(
        sourceOp->getLoc(), targetAttr.getSymbolNameFragment(), targetAttr);
    targetSymbolTable.insert(targetVariantOp);
    OpBuilder variantBuilder(&targetVariantOp.getBlock().back());
    for (auto sourceExportOp : sourceExportOps) {
      variantBuilder.clone(*sourceExportOp);

      // Map the original export names to the new variant exports.
      auto oldRefAttr = SymbolRefAttr::get(
          executableOp.getNameAttr(),
          {FlatSymbolRefAttr::get(sourceExportOp.getNameAttr())});
      auto newRefAttr = makeExportSymbolRefAttr(executableOp, targetVariantOp,
                                                sourceExportOp);
      exportExpansions[oldRefAttr].push_back(
          std::make_pair(newRefAttr, targetAttr));
    }

    // Clone any target-specific object files specified.
    if (auto objectsAttr = sourceOp.getObjectsAttr()) {
      auto objects = objectsAttr.getApplicableObjects(targetAttr);
      if (objects)
        targetVariantOp.setObjectsAttr(*objects);
    }

    // Clone inner module contents.
    if (!sourceOp.isExternal()) {
      auto sourceModuleOp = sourceOp.getInnerModule();
      variantBuilder.clone(*sourceModuleOp);
    }
  }

  // Update all dispatch sites to reference the new expanded variants.
  for (auto exportOp : sourceExportOps) {
    for (auto dispatchOp : layoutAnalysis.getExportDispatches(exportOp)) {
      updateDispatchTargets(dispatchOp, exportExpansions,
                            requiredExecutableTargets);
    }
  }

  // Remove the original.
  sourceOp.erase();
}

//===----------------------------------------------------------------------===//
// Interface definition
//===----------------------------------------------------------------------===//

// Verifies that all types used with the given entry point are supportable.
static LogicalResult
verifyEntryPointTypes(mlir::FunctionOpInterface entryFuncOp) {
  for (auto inputType : llvm::enumerate(entryFuncOp.getArgumentTypes())) {
    if (llvm::isa<IREE::Stream::BindingType>(inputType.value()) ||
        inputType.value().isInteger(32)) {
      // OK - directly translates to a HAL interface binding.
    } else {
      return entryFuncOp.emitError()
             << "unsupported interface function argument " << inputType.index()
             << " type " << inputType.value()
             << "; requires !stream.binding or i32 operands only";
    }
  }
  return success();
}

// Creates an pipeline layout attr from the analysis results.
static IREE::HAL::PipelineLayoutAttr
makePipelineLayoutAttr(const PipelineLayout &pipelineLayout,
                       IREE::HAL::ExecutableTargetAttr targetAttr,
                       OpBuilder &builder) {
  SmallVector<IREE::HAL::DescriptorSetLayoutAttr> setLayoutAttrs;
  for (const auto &setLayout : pipelineLayout.setLayouts) {
    SmallVector<IREE::HAL::DescriptorSetBindingAttr> bindingAttrs;
    for (const auto &binding : setLayout.bindings) {
      bindingAttrs.push_back(IREE::HAL::DescriptorSetBindingAttr::get(
          builder.getContext(), binding.ordinal, binding.type,
          binding.flags != IREE::HAL::DescriptorFlags::None
              ? binding.flags
              : std::optional<IREE::HAL::DescriptorFlags>{}));
    }
    std::optional<IREE::HAL::DescriptorSetLayoutFlags> flags;
    if (targetAttr.hasConfigurationAttr("hal.bindings.indirect")) {
      flags = IREE::HAL::DescriptorSetLayoutFlags::Indirect;
    }
    setLayoutAttrs.push_back(IREE::HAL::DescriptorSetLayoutAttr::get(
        builder.getContext(), setLayout.ordinal, bindingAttrs, flags));
  }
  return IREE::HAL::PipelineLayoutAttr::get(
      builder.getContext(), pipelineLayout.pushConstantCount, setLayoutAttrs);
}

// Converts the usage of the given primitive |arg| to interface methods.
static void convertOperandUsage(mlir::FunctionOpInterface sourceFuncOp,
                                BlockArgument arg, unsigned pushConstantIdx,
                                OpBuilder &builder) {
  auto alignmentAttr = sourceFuncOp.getArgAttrOfType<IntegerAttr>(
      arg.getArgNumber(), "stream.alignment");
  auto valuesAttr = sourceFuncOp.getArgAttrOfType<ArrayAttr>(arg.getArgNumber(),
                                                             "stream.values");
  auto loadOp = builder.create<IREE::HAL::InterfaceConstantLoadOp>(
      arg.getLoc(), arg.getType(), builder.getIndexAttr(pushConstantIdx),
      alignmentAttr, valuesAttr);
  arg.replaceAllUsesWith(loadOp);
}

// Converts the usage of the given !stream.binding |arg| to interface methods.
static void
convertBindingUsage(mlir::FunctionOpInterface sourceFuncOp, BlockArgument arg,
                    IREE::HAL::DescriptorSetLayoutAttr setLayoutAttr,
                    IREE::HAL::DescriptorSetBindingAttr bindingAttr) {
  if (arg.use_empty())
    return; // no-op
  for (auto &use : llvm::make_early_inc_range(arg.getUses())) {
    auto oldOp = dyn_cast<IREE::Stream::BindingSubspanOp>(use.getOwner());
    assert(oldOp && "bindings are only usable by stream.binding.subspan");
    OpBuilder builder(oldOp);
    auto alignmentAttr = sourceFuncOp.getArgAttrOfType<IntegerAttr>(
        arg.getArgNumber(), "stream.alignment");
    auto newOp = builder.create<IREE::HAL::InterfaceBindingSubspanOp>(
        oldOp.getLoc(), oldOp.getType(), APInt(64, setLayoutAttr.getOrdinal()),
        APInt(64, bindingAttr.getOrdinal()), bindingAttr.getType(),
        oldOp.getByteOffset(), oldOp.getDynamicDims(), alignmentAttr,
        bindingAttr.getFlags());
    oldOp.replaceAllUsesWith(newOp.getResult());
    oldOp.erase();
  }
}

// Clones |sourceFuncOp| and updates its signature to match the |interfaceOp|
// and use the HAL interface access primitives.
static mlir::func::FuncOp
cloneFuncWithInterface(mlir::func::FuncOp sourceFuncOp,
                       const PipelineResourceMap &resourceMap,
                       IREE::HAL::PipelineLayoutAttr layoutAttr) {
  // Clone so that we can do a bunch of unsafe in-place updates.
  auto clonedFuncOp = sourceFuncOp.clone();

  // Strip all arguments as functions take all I/O through the interface API.
  clonedFuncOp.setType(FunctionType::get(clonedFuncOp.getContext(), {}, {}));

  auto *entryBlock = &clonedFuncOp.front();
  auto entryBuilder = OpBuilder::atBlockBegin(entryBlock);

  // Change the interface from arguments to hal.interface.* methods.
  // We do push constant compatible operands first so that they are available
  // for use by the binding accessors.
  unsigned operandIdx = 0;
  for (auto arg : entryBlock->getArguments()) {
    if (!llvm::isa<IREE::Stream::BindingType>(arg.getType())) {
      convertOperandUsage(sourceFuncOp, arg, operandIdx++, entryBuilder);
    }
  }
  unsigned resourceIdx = 0;
  for (auto arg : entryBlock->getArguments()) {
    if (!llvm::isa<IREE::Stream::BindingType>(arg.getType()))
      continue;
    auto setBinding = resourceMap[resourceIdx++];
    auto setLayoutAttr = layoutAttr.getSetLayouts()[setBinding.first];
    auto bindingAttr = setLayoutAttr.getBindings()[setBinding.second];
    convertBindingUsage(sourceFuncOp, arg, setLayoutAttr, bindingAttr);
  }

  // Remove all arguments now that we've turned them into lookup ops.
  entryBlock->eraseArguments([](auto arg) { return true; });

  return clonedFuncOp;
}

// Adds the entry point ops with assigned ordinals for each entry function.
// The entry points will all use the provided |interfaceOp| and be exported with
// hal.executable.export ops.
static LogicalResult
declareEntryPointOps(IREE::Stream::ExecutableOp sourceExecutableOp,
                     IREE::HAL::ExecutableOp targetExecutableOp,
                     const BindingLayoutAnalysis &layoutAnalysis,
                     RequiredExecutableTargets &requiredExecutableTargets) {
  auto variantOps =
      targetExecutableOp.getBlock().getOps<IREE::HAL::ExecutableVariantOp>();
  OpBuilder executableBuilder(&targetExecutableOp.getBlock().front());

  // Build a map of source function definitions to their version with the
  // updated interface per variant.
  DenseMap<Operation *, DenseMap<IREE::HAL::ExecutableVariantOp, Operation *>>
      targetFuncOps;
  int nextOrdinal = 0;
  for (auto exportOp : sourceExecutableOp.getBody()
                           .getOps<IREE::Stream::ExecutableExportOp>()) {
    func::FuncOp sourceFuncOp; // optional, may be extern
    if (auto sourceModuleOp = sourceExecutableOp.getInnerModule()) {
      sourceFuncOp = sourceModuleOp.lookupSymbol<mlir::func::FuncOp>(
          exportOp.getFunctionRef());
      if (failed(verifyEntryPointTypes(sourceFuncOp)))
        return failure();
    }

    // Lookup to see if a layout was specified already. If not we'll perform
    // some basic analysis to come up with our own layout.
    auto forcedLayoutAttr =
        exportOp->getAttrOfType<IREE::HAL::PipelineLayoutAttr>(
            "hal.interface.layout");
    const auto &pipelineLayout = layoutAnalysis.getPipelineLayout(exportOp);
    const PipelineResourceMap &resourceMap = pipelineLayout.resourceMap;

    // Clone the updated function declaration into each variant.
    ExportExpansions exportExpansions;
    int ordinal = nextOrdinal++;
    for (auto variantOp : variantOps) {
      auto targetBuilder = OpBuilder::atBlockBegin(&variantOp.getBlock());

      // TODO(ravishankarm): use hal.interface.workgroup_size instead of codegen
      // attributes.
      // Check if workgroup size is set externally.
      ArrayAttr workgroupSize;
      for (auto attr : exportOp->getAttrs()) {
        if (attr.getValue().isa<IREE::Codegen::ExportConfigAttr>()) {
          workgroupSize = attr.getValue()
                              .cast<IREE::Codegen::ExportConfigAttr>()
                              .getWorkgroupSizeIndexArray();
          if (workgroupSize.size() < 3) {
            SmallVector<Attribute> workgroupSizeVals =
                llvm::to_vector(workgroupSize);
            workgroupSizeVals.resize(3, targetBuilder.getIndexAttr(1));
            workgroupSize = targetBuilder.getArrayAttr(workgroupSizeVals);
          }
          break;
        }
      }

      // Declare the entry point on the target.
      auto variantLayoutAttr =
          forcedLayoutAttr ? forcedLayoutAttr
                           : makePipelineLayoutAttr(pipelineLayout,
                                                    variantOp.getTargetAttr(),
                                                    targetBuilder);
      auto newExportOp = targetBuilder.create<IREE::HAL::ExecutableExportOp>(
          exportOp.getLoc(),
          targetBuilder.getStringAttr(exportOp.getFunctionRef()),
          targetBuilder.getIndexAttr(ordinal), variantLayoutAttr, workgroupSize,
          /*subgroup_size=*/IntegerAttr{},
          /*workgroup_local_memory=*/IntegerAttr{});

      // Map the original export name to the new variant export.
      auto oldRefAttr =
          SymbolRefAttr::get(sourceExecutableOp.getNameAttr(),
                             {FlatSymbolRefAttr::get(exportOp.getNameAttr())});
      auto newRefAttr =
          makeExportSymbolRefAttr(targetExecutableOp, variantOp, newExportOp);
      exportExpansions[oldRefAttr].push_back(
          std::make_pair(newRefAttr, variantOp.getTargetAttr()));

      // Annotate the export with the a mapping of the resources to the
      // interface bindings. This is used during conversion.
      SmallVector<Attribute> bindingAttrs;
      for (auto setBinding : resourceMap) {
        bindingAttrs.push_back(IREE::HAL::InterfaceBindingAttr::get(
            newExportOp.getContext(), setBinding.first, setBinding.second));
      }
      newExportOp->setAttr(
          "hal.interface.bindings",
          ArrayAttr::get(newExportOp.getContext(), bindingAttrs));

      // Clone the workgroup count calculation function.
      if (!exportOp.getWorkgroupCount().empty()) {
        mlir::IRMapping mapper;
        exportOp.getWorkgroupCount().cloneInto(&newExportOp.getWorkgroupCount(),
                                               mapper);
        // Insert the !hal.device argument if it doesn't already exist.
        Type deviceType = targetBuilder.getType<IREE::HAL::DeviceType>();
        if (!llvm::is_contained(exportOp.getWorkgroupCount().getArgumentTypes(),
                                deviceType)) {
          newExportOp.getWorkgroupCount().insertArgument(0u, deviceType,
                                                         newExportOp.getLoc());
        }
      }

      // Clone the source function and update it to use the new interface.
      if (sourceFuncOp) {
        auto variantFuncOp = cloneFuncWithInterface(sourceFuncOp, resourceMap,
                                                    variantLayoutAttr);
        targetFuncOps[sourceFuncOp][variantOp] = variantFuncOp;
      }
    }

    // Update all dispatch sites to reference the new expanded variants.
    for (auto dispatchOp : layoutAnalysis.getExportDispatches(exportOp)) {
      updateDispatchTargets(dispatchOp, exportExpansions,
                            requiredExecutableTargets);
    }
  }

  // Clone all of the ops in the source module to each variant.
  // We'll use the exported functions with the updated interfaces in place of
  // the original versions and copy everything else verbatim.
  // Note that we do this as a cleanup setup because there may be multiple
  // functions and multiple exports (with an N:M mapping) and in this way we
  // perform the variant construction in a single pass with deterministic
  // ordering that preserves the unmodified ops.
  if (auto sourceModuleOp = sourceExecutableOp.getInnerModule()) {
    for (auto variantOp : variantOps) {
      auto targetBuilder = OpBuilder::atBlockBegin(
          &variantOp.getInnerModule().getBodyRegion().front());
      for (auto &op : sourceModuleOp.getOps()) {
        auto targetVariantFuncOps = targetFuncOps.find(&op);
        if (targetVariantFuncOps != targetFuncOps.end()) {
          // Move the updated function into place.
          auto variantFuncOp = targetVariantFuncOps->second[variantOp];
          targetBuilder.insert(variantFuncOp);
        } else {
          // Regular op (globals, external function declarations, etc).
          targetBuilder.clone(op);
        }
      }
    }
  }

  return success();
}

//===----------------------------------------------------------------------===//
// flow.dispatch.* info op conversion
//===----------------------------------------------------------------------===//

namespace {

struct ConvertReturnPattern : public OpRewritePattern<IREE::Stream::ReturnOp> {
  using OpRewritePattern<IREE::Stream::ReturnOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(IREE::Stream::ReturnOp op,
                                PatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<IREE::HAL::ReturnOp>(op, op.getOperands());
    return success();
  }
};

template <typename SrcOp, typename DstOp>
struct ConvertDispatchWorkgroupInfoPattern final
    : public OpRewritePattern<SrcOp> {
  using OpRewritePattern<SrcOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(SrcOp op,
                                PatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<DstOp>(op, op.getResult().getType(),
                                       op.getDimensionAttr());
    return success();
  }
};

struct InlineConstantWorkgroupSizePattern
    : public OpRewritePattern<IREE::HAL::InterfaceWorkgroupSizeOp> {
  using OpRewritePattern<IREE::HAL::InterfaceWorkgroupSizeOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(IREE::HAL::InterfaceWorkgroupSizeOp sizeOp,
                                PatternRewriter &rewriter) const override {
    // Lookup the entry point matching the parent.
    auto funcOp = sizeOp->getParentOfType<mlir::func::FuncOp>();
    auto variantOp = funcOp->getParentOfType<IREE::HAL::ExecutableVariantOp>();
    auto exportOp = dyn_cast<IREE::HAL::ExecutableExportOp>(
        SymbolTable::lookupSymbolIn(variantOp, funcOp.getName()));
    assert(exportOp &&
           "must have an entry point corresponding to the parent func");
    auto workgroupSizeAttr = exportOp.getWorkgroupSizeAttr();
    if (!workgroupSizeAttr)
      return failure();

    uint64_t dimIdx = sizeOp.getDimension().getZExtValue();
    auto dimAttr = workgroupSizeAttr[dimIdx];
    rewriter.replaceOpWithNewOp<arith::ConstantOp>(
        sizeOp, rewriter.getIndexType(), cast<TypedAttr>(dimAttr));
    return success();
  }
};

} // namespace

static LogicalResult
convertDispatchWorkgroupInfoOps(IREE::HAL::ExecutableOp executableOp) {
  RewritePatternSet patterns(executableOp.getContext());
  patterns.insert<
      ConvertReturnPattern,
      ConvertDispatchWorkgroupInfoPattern<IREE::Stream::DispatchWorkgroupIDOp,
                                          IREE::HAL::InterfaceWorkgroupIDOp>,
      ConvertDispatchWorkgroupInfoPattern<
          IREE::Stream::DispatchWorkgroupCountOp,
          IREE::HAL::InterfaceWorkgroupCountOp>,
      ConvertDispatchWorkgroupInfoPattern<IREE::Stream::DispatchWorkgroupSizeOp,
                                          IREE::HAL::InterfaceWorkgroupSizeOp>,
      InlineConstantWorkgroupSizePattern>(executableOp.getContext());
  return applyPatternsAndFoldGreedily(executableOp, std::move(patterns));
}

//===----------------------------------------------------------------------===//
// --iree-hal-materialize-interfaces
//===----------------------------------------------------------------------===//

struct MaterializeInterfacesPass
    : public IREE::HAL::impl::MaterializeInterfacesPassBase<
          MaterializeInterfacesPass> {
  void runOnOperation() override {
    auto moduleOp = getOperation();
    SymbolTable symbolTable(moduleOp);
    BindingLayoutAnalysis layoutAnalysis(moduleOp, symbolTable);

    // Gather the required executable targets per executable and dispatch site.
    auto requiredExecutableTargets = buildRequiredExecutableTargetsMap(
        moduleOp, symbolTable, layoutAnalysis);

    // Handle any hand-authored executables; these only need variant expansion
    // and no layout analysis as the user specified the layout themselves.
    for (auto sourceOp : llvm::make_early_inc_range(
             moduleOp.getOps<IREE::HAL::ExecutableSourceOp>())) {
      materializeExecutableFromSourceOp(sourceOp, layoutAnalysis,
                                        requiredExecutableTargets);
    }

    // Processes all executables within the input module and produce the
    // output HAL ops. We should ensure all deduping is performed prior to
    // this when it's easier to diff IR and where we still have the flow
    // context.
    for (auto sourceOp : llvm::make_early_inc_range(
             moduleOp.getOps<IREE::Stream::ExecutableOp>())) {
      auto exportOps = sourceOp.getOps<IREE::Stream::ExecutableExportOp>();
      if (exportOps.empty())
        continue;

      // Gather a list of all #hal.executable.targets that we should produce
      // variants for based on the dispatches performed. Not all exports may be
      // used on any particular target but we let future DCE/pruning passes
      // remove them instead of modifying the inner modules here.
      SmallVector<IREE::HAL::ExecutableTargetAttr> targetAttrs(
          requiredExecutableTargets[sourceOp].getArrayRef());
      if (targetAttrs.empty())
        return;
      llvm::stable_sort(targetAttrs, [](auto lhs, auto rhs) {
        return lhs.getSymbolNameFragment() < rhs.getSymbolNameFragment();
      });

      // Create the op that will contain the translated executable.
      OpBuilder builder = OpBuilder::atBlockEnd(moduleOp.getBody());
      builder.setInsertionPointAfter(sourceOp);
      auto executableOp = builder.create<IREE::HAL::ExecutableOp>(
          sourceOp.getLoc(), sourceOp.getName());
      executableOp.setVisibility(sourceOp.getVisibility());

      // Materialize all of the hal.executable.variant ops for all backends we
      // are targeting.
      SymbolTable targetSymbolTable(executableOp);
      OpBuilder targetBuilder(&executableOp.getBlock().back());
      for (auto targetAttr : targetAttrs) {
        auto targetContainerOp =
            targetBuilder.create<IREE::HAL::ExecutableVariantOp>(
                sourceOp->getLoc(), targetAttr.getSymbolNameFragment(),
                targetAttr);
        setApplicableObjects(sourceOp, targetContainerOp);
        targetSymbolTable.insert(targetContainerOp);
        if (sourceOp.getInnerModule()) {
          OpBuilder containerBuilder(&targetContainerOp.getBlock().back());
          containerBuilder.create<mlir::ModuleOp>(sourceOp->getLoc());
        }
      }

      // Define interfaces for each exported function based on analysis.
      if (failed(declareEntryPointOps(sourceOp, executableOp, layoutAnalysis,
                                      requiredExecutableTargets))) {
        return signalPassFailure();
      }

      // Convert interface-related stream.dispatch.* ops to their
      // hal.interface.* versions.
      if (failed(convertDispatchWorkgroupInfoOps(executableOp))) {
        return signalPassFailure();
      }

      sourceOp.erase();
    }
  }
};

} // namespace

} // namespace mlir::iree_compiler::IREE::HAL
