// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/HAL/IR/HALDialect.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/Dialect/HAL/IR/HALTypes.h"
#include "iree/compiler/Dialect/HAL/Target/TargetRegistry.h"
#include "iree/compiler/Utils/StringUtils.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Parser/Parser.h"

// clang-format off: must be included after all LLVM/MLIR headers.
#define GET_ATTRDEF_CLASSES
#include "iree/compiler/Dialect/HAL/IR/HALAttrs.cpp.inc" // IWYU pragma: keep
#include "iree/compiler/Dialect/HAL/IR/HALEnums.cpp.inc" // IWYU pragma: keep
// clang-format on

namespace mlir::iree_compiler::IREE::HAL {

//===----------------------------------------------------------------------===//
// Utilities
//===----------------------------------------------------------------------===//

template <typename AttrType>
static LogicalResult parseEnumAttr(AsmParser &parser, StringRef attrName,
                                   AttrType &attr) {
  Attribute genericAttr;
  auto loc = parser.getCurrentLocation();
  if (failed(parser.parseAttribute(genericAttr,
                                   parser.getBuilder().getNoneType()))) {
    return parser.emitError(loc)
           << "failed to parse '" << attrName << "' enum string value";
  }
  auto stringAttr = llvm::dyn_cast<StringAttr>(genericAttr);
  if (!stringAttr) {
    return parser.emitError(loc)
           << "expected " << attrName << " attribute specified as string";
  }
  auto symbolized =
      symbolizeEnum<typename AttrType::ValueType>(stringAttr.getValue());
  if (!symbolized.hasValue()) {
    return parser.emitError(loc)
           << "failed to parse '" << attrName << "' enum value";
  }
  attr = AttrType::get(parser.getBuilder().getContext(), symbolized.getValue());
  return success();
}

template <typename AttrType>
static LogicalResult parseOptionalEnumAttr(AsmParser &parser,
                                           StringRef attrName, AttrType &attr) {
  if (succeeded(parser.parseOptionalQuestion())) {
    // Special case `?` to indicate any/none/undefined/etc.
    attr = AttrType::get(parser.getBuilder().getContext(), 0);
    return success();
  }
  return parseEnumAttr<AttrType>(parser, attrName, attr);
}

//===----------------------------------------------------------------------===//
// #hal.collective<*>
//===----------------------------------------------------------------------===//

// See the iree/hal/command_buffer.h iree_hal_collective_op_t for details.
uint32_t CollectiveAttr::getEncodedValue() const {
  union {
    uint32_t packed; // packed value
    struct {
      uint8_t kind;
      uint8_t reduction;
      uint8_t elementType;
      uint8_t reserved;
    };
  } value = {0};
  value.kind = static_cast<uint8_t>(getKind());
  value.reduction = static_cast<uint8_t>(
      getReduction().value_or(CollectiveReductionOp::None));
  value.elementType = static_cast<uint8_t>(getElementType());
  return value.packed;
}

//===----------------------------------------------------------------------===//
// #hal.device.target<*>
//===----------------------------------------------------------------------===//

// static
DeviceTargetAttr DeviceTargetAttr::get(MLIRContext *context,
                                       StringRef deviceID) {
  // TODO(benvanik): query default configuration from the target backend.
  return get(context, StringAttr::get(context, deviceID),
             DictionaryAttr::get(context), {});
}

// static
Attribute DeviceTargetAttr::parse(AsmParser &p, Type type) {
  StringAttr deviceIDAttr;
  DictionaryAttr configAttr;
  SmallVector<IREE::HAL::ExecutableTargetAttr> executableTargetAttrs;
  // `<"device-id"`
  if (failed(p.parseLess()) || failed(p.parseAttribute(deviceIDAttr))) {
    return {};
  }
  // `, `
  if (succeeded(p.parseOptionalComma())) {
    if (succeeded(p.parseOptionalLSquare())) {
      // `[targets, ...]` (optional)
      do {
        IREE::HAL::ExecutableTargetAttr executableTargetAttr;
        if (failed(p.parseAttribute(executableTargetAttr)))
          return {};
        executableTargetAttrs.push_back(executableTargetAttr);
      } while (succeeded(p.parseOptionalComma()));
      if (failed(p.parseRSquare()))
        return {};
    } else {
      // `{config dict}` (optional)
      if (failed(p.parseAttribute(configAttr)))
        return {};
      // `, [targets, ...]` (optional)
      if (succeeded(p.parseOptionalComma())) {
        if (failed(p.parseLSquare()))
          return {};
        do {
          IREE::HAL::ExecutableTargetAttr executableTargetAttr;
          if (failed(p.parseAttribute(executableTargetAttr)))
            return {};
          executableTargetAttrs.push_back(executableTargetAttr);
        } while (succeeded(p.parseOptionalComma()));
        if (failed(p.parseRSquare()))
          return {};
      }
    }
  }
  // `>`
  if (failed(p.parseGreater())) {
    return {};
  }
  return get(p.getContext(), deviceIDAttr, configAttr, executableTargetAttrs);
}

void DeviceTargetAttr::print(AsmPrinter &p) const {
  auto &os = p.getStream();
  os << "<";
  p.printAttribute(getDeviceID());
  auto configAttr = getConfiguration();
  if (configAttr && !configAttr.empty()) {
    os << ", ";
    p.printAttribute(configAttr);
  }
  auto executableTargetAttrs = getExecutableTargets();
  if (!executableTargetAttrs.empty()) {
    os << ", [";
    llvm::interleaveComma(executableTargetAttrs, os,
                          [&](auto executableTargetAttr) {
                            p.printAttribute(executableTargetAttr);
                          });
    os << "]";
  }
  os << ">";
}

std::string DeviceTargetAttr::getSymbolNameFragment() {
  return sanitizeSymbolName(getDeviceID().getValue().lower());
}

bool DeviceTargetAttr::hasConfigurationAttr(StringRef name) {
  auto configAttr = getConfiguration();
  return configAttr && configAttr.get(name);
}

// static
SmallVector<IREE::HAL::DeviceTargetAttr, 4>
DeviceTargetAttr::lookup(Operation *op) {
  auto attrId = mlir::StringAttr::get(op->getContext(), "hal.device.targets");
  while (op) {
    auto targetsAttr = op->getAttrOfType<ArrayAttr>(attrId);
    if (targetsAttr) {
      SmallVector<IREE::HAL::DeviceTargetAttr, 4> result;
      for (auto targetAttr : targetsAttr) {
        result.push_back(llvm::cast<IREE::HAL::DeviceTargetAttr>(targetAttr));
      }
      return result;
    }
    op = op->getParentOp();
  }
  return {}; // No devices found; let caller decide what to do.
}

// Returns a set of all configuration attributes from all device targets with
// a configuration set. Targets with no configuration set are ignored.
static SmallVector<DictionaryAttr> lookupOptionalConfigAttrs(Operation *op) {
  auto targetAttrs = IREE::HAL::DeviceTargetAttr::lookup(op);
  if (targetAttrs.empty())
    return {};
  SmallVector<DictionaryAttr> configAttrs;
  for (auto targetAttr : targetAttrs) {
    auto configAttr = targetAttr.getConfiguration();
    if (configAttr)
      configAttrs.push_back(configAttr);
  }
  return configAttrs;
}

// Returns a set of all configuration attributes from all device targets.
// Returns nullopt if any target is missing a configuration attribute.
static std::optional<SmallVector<DictionaryAttr>>
lookupRequiredConfigAttrs(Operation *op) {
  auto targetAttrs = IREE::HAL::DeviceTargetAttr::lookup(op);
  if (targetAttrs.empty())
    return std::nullopt;
  SmallVector<DictionaryAttr> configAttrs;
  for (auto targetAttr : targetAttrs) {
    auto configAttr = targetAttr.getConfiguration();
    if (!configAttr)
      return std::nullopt;
    configAttrs.push_back(configAttr);
  }
  return configAttrs;
}

template <typename AttrT>
static std::optional<typename AttrT::ValueType> joinConfigAttrs(
    ArrayRef<DictionaryAttr> configAttrs, StringRef name,
    std::function<typename AttrT::ValueType(typename AttrT::ValueType,
                                            typename AttrT::ValueType)>
        join) {
  if (configAttrs.empty())
    return std::nullopt;
  auto firstValue = configAttrs.front().getAs<AttrT>(name);
  if (!firstValue)
    return std::nullopt;
  auto result = firstValue.getValue();
  for (auto configAttr : configAttrs.drop_front(1)) {
    auto value = configAttr.getAs<AttrT>(name);
    if (!value)
      return std::nullopt;
    result = join(result, value.getValue());
  }
  return result;
}

template <typename AttrT>
static std::optional<StaticRange<typename AttrT::ValueType>>
joinConfigStaticRanges(ArrayRef<DictionaryAttr> configAttrs, StringRef name,
                       std::function<StaticRange<typename AttrT::ValueType>(
                           StaticRange<typename AttrT::ValueType>,
                           StaticRange<typename AttrT::ValueType>)>
                           join) {
  if (configAttrs.empty())
    return std::nullopt;
  auto firstValue = configAttrs.front().getAs<AttrT>(name);
  if (!firstValue)
    return std::nullopt;
  StaticRange<typename AttrT::ValueType> result{firstValue.getValue()};
  for (auto configAttr : configAttrs.drop_front(1)) {
    auto value = configAttr.getAs<AttrT>(name);
    if (!value)
      return std::nullopt;
    result =
        join(result, StaticRange<typename AttrT::ValueType>{value.getValue()});
  }
  return result;
}

// static
bool DeviceTargetAttr::lookupConfigAttrAny(Operation *op, StringRef name) {
  auto configAttrs = lookupOptionalConfigAttrs(op);
  if (configAttrs.empty())
    return false;
  for (auto configAttr : configAttrs) {
    if (configAttr.get(name))
      return true;
  }
  return false;
}

// static
bool DeviceTargetAttr::lookupConfigAttrAll(Operation *op, StringRef name) {
  auto configAttrs = lookupRequiredConfigAttrs(op);
  if (!configAttrs)
    return false;
  for (auto configAttr : *configAttrs) {
    if (!configAttr.get(name))
      return false;
  }
  return true;
}

// static
std::optional<bool> DeviceTargetAttr::lookupConfigAttrAnd(Operation *op,
                                                          StringRef name) {
  auto configAttrs = lookupRequiredConfigAttrs(op);
  if (!configAttrs)
    return std::nullopt;
  return joinConfigAttrs<BoolAttr>(
      configAttrs.value(), name, [](bool lhs, bool rhs) { return lhs && rhs; });
}

// static
std::optional<bool> DeviceTargetAttr::lookupConfigAttrOr(Operation *op,
                                                         StringRef name) {
  auto configAttrs = lookupRequiredConfigAttrs(op);
  if (!configAttrs)
    return std::nullopt;
  return joinConfigAttrs<BoolAttr>(
      configAttrs.value(), name, [](bool lhs, bool rhs) { return lhs || rhs; });
}

// static
std::optional<StaticRange<APInt>>
DeviceTargetAttr::lookupConfigAttrRange(Operation *op, StringRef name) {
  auto configAttrs = lookupRequiredConfigAttrs(op);
  if (!configAttrs)
    return std::nullopt;
  return joinConfigStaticRanges<IntegerAttr>(
      configAttrs.value(), name,
      [](StaticRange<APInt> lhs, StaticRange<APInt> rhs) {
        return StaticRange<APInt>{
            llvm::APIntOps::smin(lhs.min, rhs.min),
            llvm::APIntOps::smax(lhs.max, rhs.max),
        };
      });
}

// static
SmallVector<ExecutableTargetAttr, 4>
DeviceTargetAttr::lookupExecutableTargets(Operation *op) {
  SmallVector<ExecutableTargetAttr, 4> resultAttrs;
  for (auto deviceTargetAttr : lookup(op)) {
    for (auto executableTargetAttr : deviceTargetAttr.getExecutableTargets()) {
      if (!llvm::is_contained(resultAttrs, executableTargetAttr)) {
        resultAttrs.push_back(executableTargetAttr);
      }
    }
  }
  return resultAttrs;
}

void IREE::HAL::DeviceTargetAttr::printStatusDescription(
    llvm::raw_ostream &os) const {
  cast<Attribute>().print(os, /*elideType=*/true);
}

// Produces a while-loop that enumerates each device available and tries to
// match it against the target information. SCF is... not very wieldy, but this
// is effectively:
// ```
//   %device_count = hal.devices.count : index
//   %result:2 = scf.while(%i = 0, %device = null) {
//     %is_null = util.cmp.eq %device, null : !hal.device
//     %in_bounds = arith.cmpi slt %i, %device_count : index
//     %continue_while = arith.andi %is_null, %in_bounds : i1
//     scf.condition(%continue_while) %i, %device : index, !hal.device
//   } do {
//     %device_i = hal.devices.get %i : !hal.device
//     %is_match = <<buildDeviceMatch>>(%device_i)
//     %try_device = arith.select %is_match, %device_i, null : !hal.device
//     %next_i = arith.addi %i, %c1 : index
//     scf.yield %next_i, %try_device : index, !hal.device
//   }
// ```
// Upon completion %result#1 contains the device (or null).
Value IREE::HAL::DeviceTargetAttr::buildDeviceEnumeration(
    Location loc, const IREE::HAL::TargetRegistry &targetRegistry,
    OpBuilder &builder) const {
  // Defers to the target backend to build the device match or does a simple
  // fallback for unregistered backends (usually for testing, but may be used
  // as a way to bypass validation for out-of-tree experiments).
  auto buildDeviceMatch = [&](Location loc, Value device,
                              OpBuilder &builder) -> Value {
    // Ask the target backend to build the match expression. It may opt to
    // let the default handling take care of things.
    Value match;
    auto targetDevice = targetRegistry.getTargetDevice(getDeviceID());
    if (targetDevice) {
      match = targetDevice->buildDeviceTargetMatch(loc, device, *this, builder);
    }
    if (match)
      return match;

    // Match first on the device ID, as that's the top-level filter.
    Value idMatch = IREE::HAL::DeviceQueryOp::createI1(
        loc, device, "hal.device.id", getDeviceID(), builder);

    // If there are executable formats defined we should check at least one of
    // them is supported.
    auto executableTargetAttrs =
        const_cast<IREE::HAL::DeviceTargetAttr *>(this)->getExecutableTargets();
    if (executableTargetAttrs.empty()) {
      match = idMatch; // just device ID
    } else {
      auto ifOp = builder.create<scf::IfOp>(loc, builder.getI1Type(), idMatch,
                                            true, true);
      auto thenBuilder = ifOp.getThenBodyBuilder();
      Value anyFormatMatch;
      for (auto executableTargetAttr : executableTargetAttrs) {
        Value formatMatch = IREE::HAL::DeviceQueryOp::createI1(
            loc, device, "hal.executable.format",
            executableTargetAttr.getFormat().getValue(), thenBuilder);
        if (!anyFormatMatch) {
          anyFormatMatch = formatMatch;
        } else {
          anyFormatMatch = thenBuilder.create<arith::OrIOp>(loc, anyFormatMatch,
                                                            formatMatch);
        }
      }
      thenBuilder.create<scf::YieldOp>(loc, anyFormatMatch);
      auto elseBuilder = ifOp.getElseBodyBuilder();
      Value falseValue = elseBuilder.create<arith::ConstantIntOp>(loc, 0, 1);
      elseBuilder.create<scf::YieldOp>(loc, falseValue);
      match = ifOp.getResult(0);
    }

    return match;
  };

  // Enumerate all devices and match the first one found (if any).
  Type indexType = builder.getIndexType();
  Type deviceType = builder.getType<IREE::HAL::DeviceType>();
  Value c0 = builder.create<arith::ConstantIndexOp>(loc, 0);
  Value c1 = builder.create<arith::ConstantIndexOp>(loc, 1);
  Value nullDevice = builder.create<IREE::Util::NullOp>(loc, deviceType);
  Value deviceCount = builder.create<IREE::HAL::DevicesCountOp>(loc, indexType);
  auto whileOp = builder.create<scf::WhileOp>(
      loc, TypeRange{indexType, deviceType}, ValueRange{c0, nullDevice},
      [&](OpBuilder &beforeBuilder, Location loc, ValueRange operands) {
        Value isNull = beforeBuilder.create<IREE::Util::CmpEQOp>(
            loc, operands[1], nullDevice);
        Value inBounds = beforeBuilder.create<arith::CmpIOp>(
            loc, arith::CmpIPredicate::slt, operands[0], deviceCount);
        Value continueWhile =
            beforeBuilder.create<arith::AndIOp>(loc, isNull, inBounds);
        beforeBuilder.create<scf::ConditionOp>(loc, continueWhile, operands);
      },
      [&](OpBuilder &afterBuilder, Location loc, ValueRange operands) {
        Value device = afterBuilder.create<IREE::HAL::DevicesGetOp>(
            loc, deviceType, operands[0]);
        Value isMatch = buildDeviceMatch(loc, device, afterBuilder);
        Value tryDevice = afterBuilder.create<arith::SelectOp>(
            loc, isMatch, device, nullDevice);
        Value nextI = afterBuilder.create<arith::AddIOp>(loc, operands[0], c1);
        afterBuilder.create<scf::YieldOp>(loc, ValueRange{nextI, tryDevice});
      });
  return whileOp.getResult(1);
}

//===----------------------------------------------------------------------===//
// #hal.executable.target<*>
//===----------------------------------------------------------------------===//

// static
ExecutableTargetAttr ExecutableTargetAttr::get(MLIRContext *context,
                                               StringRef backend,
                                               StringRef format) {
  return get(context, StringAttr::get(context, backend),
             StringAttr::get(context, format), DictionaryAttr::get(context));
}

// static
Attribute ExecutableTargetAttr::parse(AsmParser &p, Type type) {
  StringAttr backendAttr;
  StringAttr formatAttr;
  DictionaryAttr configurationAttr;
  // `<"backend", "format"`
  if (failed(p.parseLess()) || failed(p.parseAttribute(backendAttr)) ||
      failed(p.parseComma()) || failed(p.parseAttribute(formatAttr))) {
    return {};
  }
  // `, {config}`
  if (succeeded(p.parseOptionalComma()) &&
      failed(p.parseAttribute(configurationAttr))) {
    return {};
  }
  // `>`
  if (failed(p.parseGreater())) {
    return {};
  }
  return get(p.getContext(), backendAttr, formatAttr, configurationAttr);
}

void ExecutableTargetAttr::print(AsmPrinter &p) const {
  auto &os = p.getStream();
  os << "<";
  p.printAttribute(getBackend());
  os << ", ";
  p.printAttribute(getFormat());
  auto config = getConfiguration();
  if (config && !config.empty()) {
    os << ", ";
    p.printAttribute(config);
  }
  os << ">";
}

std::string ExecutableTargetAttr::getSymbolNameFragment() const {
  return sanitizeSymbolName(getFormat().getValue().lower());
}

bool ExecutableTargetAttr::hasConfigurationAttr(StringRef name) {
  auto configAttr = getConfiguration();
  return configAttr && configAttr.get(name);
}

// For now this is very simple: if there are any specified fields that are
// present in this attribute they must match. We could allow target backends
// to customize this via attribute interfaces in the future if we needed.
bool ExecutableTargetAttr::isGenericOf(
    IREE::HAL::ExecutableTargetAttr specificAttr) {
  if (getBackend() != specificAttr.getBackend() ||
      getFormat() != specificAttr.getFormat()) {
    // Totally different backends and binary formats.
    // There may be cases where we want to share things - such as when targeting
    // both DLLs and dylibs or something - but today almost all of these are
    // unique situations.
    return false;
  }

  // If the config is empty on either we can quickly match.
  // This is the most common case for users manually specifying targets.
  auto genericConfigAttr = getConfiguration();
  auto specificConfigAttr = specificAttr.getConfiguration();
  if (!genericConfigAttr || !specificConfigAttr)
    return true;

  // Ensure all fields in specificConfigAttr either don't exist or match.
  for (auto expectedAttr : specificConfigAttr.getValue()) {
    auto actualValue = genericConfigAttr.getNamed(expectedAttr.getName());
    if (!actualValue) {
      continue; // ignore, not present in generic
    }
    if (actualValue->getValue() != expectedAttr.getValue()) {
      return false; // mismatch, both have values but they differ
    }
  }

  // Ensure all fields in genericConfigAttr exist in the specific one.
  // If missing then the generic is _more_ specific and can't match.
  for (auto actualAttr : genericConfigAttr.getValue()) {
    if (!specificConfigAttr.getNamed(actualAttr.getName())) {
      return false; // mismatch, present in generic but not specific
    }
  }

  // All fields match or are omitted in the generic version.
  return true;
}

// static
ExecutableTargetAttr ExecutableTargetAttr::lookup(Operation *op) {
  auto *context = op->getContext();
  auto attrId = StringAttr::get(context, "hal.executable.target");
  while (op) {
    // Take directly from the enclosing variant.
    if (auto variantOp = llvm::dyn_cast<IREE::HAL::ExecutableVariantOp>(op)) {
      return variantOp.getTarget();
    }
    // Use an override if specified.
    auto attr = op->getAttrOfType<IREE::HAL::ExecutableTargetAttr>(attrId);
    if (attr)
      return attr;
    // Continue walk.
    op = op->getParentOp();
  }
  // No target found during walk. No default to provide so fail and let the
  // caller decide what to do (assert/fallback/etc).
  return nullptr;
}

//===----------------------------------------------------------------------===//
// #hal.executable.object<*>
//===----------------------------------------------------------------------===//

// static
Attribute ExecutableObjectAttr::parse(AsmParser &p, Type type) {
  NamedAttrList dict;
  // `<{` dict `}>`
  if (failed(p.parseLess()) || failed(p.parseOptionalAttrDict(dict)) ||
      failed(p.parseGreater())) {
    return {};
  }
  auto pathAttr = llvm::dyn_cast_if_present<StringAttr>(dict.get("path"));
  auto dataAttr =
      llvm::dyn_cast_if_present<DenseIntElementsAttr>(dict.get("data"));
  return get(p.getContext(), pathAttr, dataAttr);
}

void ExecutableObjectAttr::print(AsmPrinter &p) const {
  auto &os = p.getStream();
  os << "<{";
  if (auto pathAttr = getPath()) {
    os << "path = ";
    p.printAttribute(getPath());
  }
  if (auto dataAttr = getData()) {
    os << ", data = ";
    p.printAttribute(getData());
  }
  os << "}>";
}

// static
void ExecutableObjectAttr::filterObjects(
    ArrayAttr objectAttrs, ArrayRef<StringRef> extensions,
    SmallVectorImpl<IREE::HAL::ExecutableObjectAttr> &filteredAttrs) {
  if (!objectAttrs)
    return;
  for (auto objectAttr :
       objectAttrs.getAsRange<IREE::HAL::ExecutableObjectAttr>()) {
    auto path = objectAttr.getPath();
    auto ext = llvm::sys::path::extension(path);
    if (llvm::is_contained(extensions, ext)) {
      filteredAttrs.push_back(objectAttr);
    }
  }
}

// Tries to find |filePath| on disk either at its absolute path or joined with
// any of the specified |searchPaths| in order.
// Returns the absolute file path when found or a failure if there are no hits.
static FailureOr<std::string>
findFileInPaths(StringRef filePath, ArrayRef<std::string> searchPaths) {
  // First try to see if it's an absolute path - we don't want to perform any
  // additional processing on top of that.
  if (llvm::sys::path::is_absolute(filePath)) {
    if (llvm::sys::fs::exists(filePath))
      return filePath.str();
    return failure();
  }

  // Try a relative lookup from the current working directory.
  if (llvm::sys::fs::exists(filePath))
    return filePath.str();

  // Search each path in turn for a file that exists.
  // It doesn't mean we can open it but we'll get a better error out of the
  // actual open attempt than what we could produce here.
  for (auto searchPath : searchPaths) {
    SmallVector<char> tryPath{searchPath.begin(), searchPath.end()};
    llvm::sys::path::append(tryPath, filePath);
    if (llvm::sys::fs::exists(Twine(tryPath)))
      return Twine(tryPath).str();
  }

  // Not found in either the user-specified absolute path, cwd, or the search
  // paths.
  return failure();
}

static llvm::cl::list<std::string> clExecutableObjectSearchPath(
    "iree-hal-executable-object-search-path",
    llvm::cl::desc("Additional search paths for resolving "
                   "#hal.executable.object file references."),
    llvm::cl::ZeroOrMore);

FailureOr<std::string> ExecutableObjectAttr::getAbsolutePath() {
  auto pathAttr = getPath();
  if (!pathAttr)
    return failure(); // not a file reference
  return findFileInPaths(pathAttr.getValue(), clExecutableObjectSearchPath);
}

std::optional<std::string> ExecutableObjectAttr::loadData() {
  if (auto dataAttr = getData()) {
    // This is shady but so is using this feature.
    // TODO(benvanik): figure out a way to limit the attribute to signless int8.
    // We could share the attribute -> byte array code with the VM constant
    // serialization if we wanted.
    auto rawData = dataAttr.getRawData();
    return std::string(rawData.data(), rawData.size());
  } else if (auto pathAttr = getPath()) {
    // Search for file and try to load it if found.
    auto filePath =
        findFileInPaths(pathAttr.getValue(), clExecutableObjectSearchPath);
    if (failed(filePath)) {
      llvm::errs()
          << "ERROR: referenced object file not found on any path; use "
             "--iree-hal-executable-object-search-path= to add search paths: "
          << *this << "\n";
      return std::nullopt;
    }
    auto file = llvm::MemoryBuffer::getFile(*filePath);
    if (!file)
      return std::nullopt;
    return std::string((*file)->getBuffer());
  }
  return std::nullopt;
}

//===----------------------------------------------------------------------===//
// #hal.executable.objects<*>
//===----------------------------------------------------------------------===//

// static
LogicalResult ExecutableObjectsAttr::verify(
    function_ref<mlir::InFlightDiagnostic()> emitError, ArrayAttr targetsAttr,
    ArrayAttr targetObjectsAttr) {
  if (targetsAttr.size() != targetObjectsAttr.size()) {
    return emitError() << "targets and objects must be 1:1";
  }
  for (auto targetAttr : targetsAttr) {
    if (!llvm::isa<IREE::HAL::ExecutableTargetAttr>(targetAttr)) {
      return emitError()
             << "target keys must be #hal.executable.target attributes";
    }
  }
  for (auto objectsAttr : targetObjectsAttr) {
    auto objectsArrayAttr = llvm::dyn_cast<ArrayAttr>(objectsAttr);
    if (!objectsArrayAttr) {
      return emitError() << "target objects must be an array of "
                            "#hal.executable.object attributes";
    }
  }
  return success();
}

// static
Attribute ExecutableObjectsAttr::parse(AsmParser &p, Type type) {
  // `<{` target = [objects, ...], ... `}>`
  SmallVector<Attribute> targetAttrs;
  SmallVector<Attribute> objectsAttrs;
  if (failed(p.parseLess()))
    return {};
  if (succeeded(p.parseLBrace()) && !succeeded(p.parseOptionalRBrace())) {
    do {
      Attribute targetAttr;
      ArrayAttr objectsAttr;
      if (failed(p.parseAttribute(targetAttr)) || failed(p.parseEqual()) ||
          failed(p.parseAttribute(objectsAttr))) {
        return {};
      }
      targetAttrs.push_back(targetAttr);
      objectsAttrs.push_back(objectsAttr);
    } while (succeeded(p.parseOptionalComma()));
    if (failed(p.parseRBrace()))
      return {};
  }
  if (failed(p.parseGreater()))
    return {};
  return get(p.getContext(), ArrayAttr::get(p.getContext(), targetAttrs),
             ArrayAttr::get(p.getContext(), objectsAttrs));
}

void ExecutableObjectsAttr::print(AsmPrinter &p) const {
  auto &os = p.getStream();
  os << "<{";
  llvm::interleaveComma(llvm::zip_equal(getTargets(), getTargetObjects()), os,
                        [&](std::tuple<Attribute, Attribute> keyValue) {
                          p.printAttribute(std::get<0>(keyValue));
                          os << " = ";
                          p.printAttributeWithoutType(std::get<1>(keyValue));
                        });
  os << "}>";
}

std::optional<ArrayAttr> ExecutableObjectsAttr::getApplicableObjects(
    IREE::HAL::ExecutableTargetAttr specificTargetAttr) {
  SmallVector<Attribute> allObjectAttrs;
  for (auto [targetAttr, objectsAttr] :
       llvm::zip_equal(getTargets(), getTargetObjects())) {
    auto genericTargetAttr =
        llvm::cast<IREE::HAL::ExecutableTargetAttr>(targetAttr);
    if (genericTargetAttr.isGenericOf(specificTargetAttr)) {
      auto objectsArrayAttr = llvm::cast<ArrayAttr>(objectsAttr);
      allObjectAttrs.append(objectsArrayAttr.begin(), objectsArrayAttr.end());
    }
  }
  if (allObjectAttrs.empty())
    return std::nullopt;
  return ArrayAttr::get(specificTargetAttr.getContext(), allObjectAttrs);
}

//===----------------------------------------------------------------------===//
// #hal.device.ordinal<*>
//===----------------------------------------------------------------------===//

void IREE::HAL::DeviceOrdinalAttr::printStatusDescription(
    llvm::raw_ostream &os) const {
  cast<Attribute>().print(os, /*elideType=*/true);
}

Value IREE::HAL::DeviceOrdinalAttr::buildDeviceEnumeration(
    Location loc, const IREE::HAL::TargetRegistry &targetRegistry,
    OpBuilder &builder) const {
  return builder.create<IREE::HAL::DevicesGetOp>(
      loc, getType(),
      builder.create<arith::ConstantIndexOp>(loc, getOrdinal()));
}

//===----------------------------------------------------------------------===//
// #hal.device.fallback<*>
//===----------------------------------------------------------------------===//

void IREE::HAL::DeviceFallbackAttr::printStatusDescription(
    llvm::raw_ostream &os) const {
  cast<Attribute>().print(os, /*elideType=*/true);
}

Value IREE::HAL::DeviceFallbackAttr::buildDeviceEnumeration(
    Location loc, const IREE::HAL::TargetRegistry &targetRegistry,
    OpBuilder &builder) const {
  // TODO(benvanik): hal.device.cast if needed - may need to look up the global
  // to do it as we don't encode what the device is here in a way that is
  // guaranteed to be consistent.
  return builder.create<IREE::Util::GlobalLoadOp>(loc, getType(),
                                                  getName().getValue());
}

//===----------------------------------------------------------------------===//
// #hal.device.select<*>
//===----------------------------------------------------------------------===//

// static
LogicalResult
DeviceSelectAttr::verify(function_ref<mlir::InFlightDiagnostic()> emitError,
                         Type type, ArrayAttr devicesAttr) {
  if (devicesAttr.empty())
    return emitError() << "must have at least one device to select";
  for (auto deviceAttr : devicesAttr) {
    if (!deviceAttr.isa<IREE::HAL::DeviceInitializationAttrInterface>()) {
      return emitError() << "can only select between #hal.device.target, "
                            "#hal.device.ordinal, #hal.device.fallback, or "
                            "other device initialization attributes";
    }
  }
  // TODO(benvanik): when !hal.device is parameterized we should check that the
  // type is compatible with the entries.
  return success();
}

void IREE::HAL::DeviceSelectAttr::printStatusDescription(
    llvm::raw_ostream &os) const {
  // TODO(benvanik): print something easier to read (newline per device, etc).
  cast<Attribute>().print(os, /*elideType=*/true);
}

// Builds a recursive nest of try-else blocks for each device specified.
Value IREE::HAL::DeviceSelectAttr::buildDeviceEnumeration(
    Location loc, const IREE::HAL::TargetRegistry &targetRegistry,
    OpBuilder &builder) const {
  Type deviceType = builder.getType<IREE::HAL::DeviceType>();
  Value nullDevice = builder.create<IREE::Util::NullOp>(loc, deviceType);
  std::function<Value(ArrayRef<IREE::HAL::DeviceInitializationAttrInterface>,
                      OpBuilder &)>
      buildTry;
  buildTry =
      [&](ArrayRef<IREE::HAL::DeviceInitializationAttrInterface> deviceAttrs,
          OpBuilder &tryBuilder) -> Value {
    auto deviceAttr = deviceAttrs.front();
    Value tryDevice =
        deviceAttr.buildDeviceEnumeration(loc, targetRegistry, tryBuilder);
    if (deviceAttrs.size() == 1)
      return tryDevice; // termination case
    Value isNull =
        tryBuilder.create<IREE::Util::CmpEQOp>(loc, tryDevice, nullDevice);
    auto ifOp =
        tryBuilder.create<scf::IfOp>(loc, deviceType, isNull, true, true);
    auto thenBuilder = ifOp.getThenBodyBuilder();
    Value tryChainDevice = buildTry(deviceAttrs.drop_front(1), thenBuilder);
    thenBuilder.create<scf::YieldOp>(loc, tryChainDevice);
    auto elseBuilder = ifOp.getElseBodyBuilder();
    elseBuilder.create<scf::YieldOp>(loc, tryDevice);
    return ifOp.getResult(0);
  };
  SmallVector<IREE::HAL::DeviceInitializationAttrInterface> deviceAttrs(
      getDevices().getAsRange<IREE::HAL::DeviceInitializationAttrInterface>());
  return buildTry(deviceAttrs, builder);
}

//===----------------------------------------------------------------------===//
// #hal.affinity.queue<*>
//===----------------------------------------------------------------------===//

// static
Attribute AffinityQueueAttr::parse(AsmParser &p, Type type) {
  int64_t mask = 0;
  // `<`
  if (failed(p.parseLess()))
    return {};
  // `*` (any)
  if (succeeded(p.parseOptionalStar())) {
    mask = -1;
  } else {
    // `[`queue_bit[, ...] `]`
    if (failed(p.parseCommaSeparatedList(AsmParser::Delimiter::Square, [&]() {
          int64_t i = 0;
          if (failed(p.parseInteger(i)))
            return failure();
          mask |= 1ll << i;
          return success();
        }))) {
      return {};
    }
  }
  // `>`
  if (failed(p.parseGreater()))
    return {};
  return get(p.getContext(), mask);
}

void AffinityQueueAttr::print(AsmPrinter &p) const {
  auto &os = p.getStream();
  os << "<";
  int64_t mask = getMask();
  if (mask == -1) {
    os << "*";
  } else {
    os << "[";
    for (int i = 0, j = 0; i < sizeof(mask) * 8; ++i) {
      if (mask & (1ll << i)) {
        if (j++ > 0)
          os << ", ";
        os << i;
      }
    }
    os << "]";
  }
  os << ">";
}

bool AffinityQueueAttr::isExecutableWith(
    IREE::Stream::AffinityAttr other) const {
  if (!other)
    return true;
  // Only compatible with other queue affinities today. When we extend the
  // attributes to specify device targets we'd want to check here.
  auto otherQueueAttr = llvm::dyn_cast_if_present<AffinityQueueAttr>(other);
  if (!otherQueueAttr)
    return false;
  // If this affinity is a subset of the target affinity then it can execute
  // with it.
  if ((getMask() & otherQueueAttr.getMask()) == getMask())
    return true;
  // Otherwise not compatible.
  return false;
}

IREE::Stream::AffinityAttr
AffinityQueueAttr::joinOR(IREE::Stream::AffinityAttr other) const {
  if (!other)
    return *this;
  if (!IREE::Stream::AffinityAttr::canExecuteTogether(*this, other)) {
    return nullptr;
  }
  auto otherQueueAttr = llvm::dyn_cast_if_present<AffinityQueueAttr>(other);
  return AffinityQueueAttr::get(getContext(),
                                getMask() | otherQueueAttr.getMask());
}

IREE::Stream::AffinityAttr
AffinityQueueAttr::joinAND(IREE::Stream::AffinityAttr other) const {
  if (!other)
    return *this;
  if (!IREE::Stream::AffinityAttr::canExecuteTogether(*this, other)) {
    return nullptr;
  }
  auto otherQueueAttr = llvm::dyn_cast_if_present<AffinityQueueAttr>(other);
  return AffinityQueueAttr::get(getContext(),
                                getMask() & otherQueueAttr.getMask());
}

//===----------------------------------------------------------------------===//
// IREE::HAL::HALDialect
//===----------------------------------------------------------------------===//

// At the end so it can use functions above:
#include "iree/compiler/Dialect/HAL/IR/HALAttrInterfaces.cpp.inc"

void HALDialect::registerAttributes() {
  // Register command line flags:
  (void)clExecutableObjectSearchPath;

  addAttributes<
#define GET_ATTRDEF_LIST
#include "iree/compiler/Dialect/HAL/IR/HALAttrs.cpp.inc" // IWYU pragma: keep
      >();
}

Attribute HALDialect::parseAttribute(DialectAsmParser &parser,
                                     Type type) const {
  StringRef mnemonic;
  Attribute genAttr;
  OptionalParseResult parseResult =
      generatedAttributeParser(parser, &mnemonic, type, genAttr);
  if (parseResult.has_value())
    return genAttr;
  parser.emitError(parser.getNameLoc())
      << "unknown HAL attribute: " << mnemonic;
  return {};
}

void HALDialect::printAttribute(Attribute attr, DialectAsmPrinter &p) const {
  TypeSwitch<Attribute>(attr).Default([&](Attribute) {
    if (failed(generatedAttributePrinter(attr, p))) {
      assert(false && "unhandled HAL attribute kind");
    }
  });
}

} // namespace mlir::iree_compiler::IREE::HAL
