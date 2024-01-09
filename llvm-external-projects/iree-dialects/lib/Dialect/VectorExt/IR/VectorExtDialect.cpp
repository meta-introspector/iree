// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-dialects/Dialect/VectorExt/IR/VectorExtDialect.h"
#include "iree-dialects/Dialect/VectorExt/IR/VectorExtOps.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"
#include <numeric>

using namespace mlir;
using namespace mlir::iree_compiler::IREE::VectorExt;

#include "iree-dialects/Dialect/VectorExt/IR/VectorExtEnums.cpp.inc" // IWYU pragma: keep

#define GET_ATTRDEF_CLASSES
#include "iree-dialects/Dialect/VectorExt/IR/VectorExtAttrs.cpp.inc" // IWYU pragma: keep
                                                                     //
struct IREEVectorExtDialectOpAsmInterface : public OpAsmDialectInterface {
  using OpAsmDialectInterface::OpAsmDialectInterface;
  AliasResult getAlias(Attribute attr, raw_ostream &os) const override {
    if (llvm::isa<LayoutAttr>(attr)) {
      os << "layout";
      return AliasResult::OverridableAlias;
    }
    return AliasResult::NoAlias;
  }
};

void IREEVectorExtDialect::initialize() {
  addInterfaces<IREEVectorExtDialectOpAsmInterface>();
  addAttributes<
#define GET_ATTRDEF_LIST
#include "iree-dialects/Dialect/VectorExt/IR/VectorExtAttrs.cpp.inc"
      >();

#define GET_OP_LIST
  addOperations<
#include "iree-dialects/Dialect/VectorExt/IR/VectorExtOps.cpp.inc"
      >();
}

#include "iree-dialects/Dialect/VectorExt/IR/VectorExtDialect.cpp.inc"

#include "iree-dialects/Dialect/VectorExt/IR/VectorExtAttrInterfaces.cpp.inc"

bool PerDimLayoutAttr::contains(const LayoutDimension &dim) {
  for (LayoutDimensionAttr label : getLabels()) {
    if (label.getValue() == dim)
      return true;
  }
  return false;
}

std::optional<int64_t>
PerDimLayoutAttr::getShape(const LayoutDimension &dim) const {
  for (auto value : llvm::zip(getLabels(), getShapes())) {
    if (dim == std::get<0>(value).getValue())
      return std::get<1>(value);
  }
  return std::nullopt;
}

std::optional<int64_t> LayoutAttr::getShape(const LayoutDimension &dim) const {
  for (PerDimLayoutAttr layout : getLayouts()) {
    auto maybeShape = layout.getShape(dim);
    if (maybeShape)
      return *maybeShape;
  }
  return std::nullopt;
}

// Get the SIMT Vector shape in the order specified by dims. If no dims are
// specified, then return an empty vector.
SmallVector<int64_t>
LayoutAttr::getSIMTVectorShape(ArrayRef<LayoutDimension> dims) const {
  SmallVector<int64_t> simtVectorShape;
  for (LayoutDimension dim : dims) {
    ArrayRef<PerDimLayoutAttr> layouts = getLayouts();
    for (PerDimLayoutAttr layout : layouts) {
      if (!layout.contains(dim))
        continue;
      simtVectorShape.push_back(layout.getShape(dim).value());
    }
  }
  return simtVectorShape;
}

PerDimLayoutAttr LayoutAttr::getDimLayout(int64_t dim) const {
  assert(dim >= 0 && dim < getLayouts().size());
  return getLayouts()[dim];
}

bool LayoutAttr::isValidLayout(ArrayRef<int64_t> shape) const {
  for (auto perDimLayout : llvm::enumerate(getLayouts())) {
    ArrayRef<int64_t> layoutShape = perDimLayout.value().getShapes();
    int64_t computedShape = std::reduce(layoutShape.begin(), layoutShape.end(),
                                        1, std::multiplies<int64_t>());
    int64_t expectedShape = shape[perDimLayout.index()];
    if (computedShape != expectedShape) {
      return false;
    }
  }
  return true;
}

static int64_t getInnermostVectorShape(ArrayRef<LayoutDimensionAttr> labels,
                                       ArrayRef<int64_t> shapes) {
  return isVector(labels.back().getValue()) ? shapes.back() : 1;
}

namespace mlir::iree_compiler::IREE::VectorExt {

AffineExpr computeSIMDIndex(const LayoutIterator::State &state,
                            const PerDimLayoutAttr &attr) {
  DenseSet<LayoutDimension> layoutDims;
  for (auto label : attr.getLabels()) {
    if (isLane(label.getValue()))
      layoutDims.insert(label.getValue());
  }
  MLIRContext *ctx = attr.getContext();
  SmallVector<AffineExpr> dims(layoutDims.size());
  bindDimsList(ctx, MutableArrayRef{dims});
  AffineExpr offset = getAffineConstantExpr(0, ctx);
  AffineExpr stride = getAffineConstantExpr(1, ctx);
  int i = 0;
  for (const auto &[nameAttr, shape] : llvm::zip(
           llvm::reverse(attr.getLabels()), llvm::reverse(attr.getShapes()))) {
    LayoutDimension name = nameAttr.getValue();
    if (layoutDims.contains(name)) {
      offset = offset + stride * dims[i++];
      stride = stride * getAffineConstantExpr(shape, ctx);
      continue;
    }
    if (!state.contains(name))
      continue;
    offset = offset + stride * getAffineConstantExpr(
                                   state.lookup(name).getPosition(), ctx);
    stride = stride * getAffineConstantExpr(shape, ctx);
  }
  return offset;
}

} // namespace mlir::iree_compiler::IREE::VectorExt

// Get the offset into the SIMT vector corresponding to the incoming iterator.
// The returned offsets will always be the same shape as the labels array.
SmallVector<int64_t> LayoutIterator::State::computeSIMTIndex(
    ArrayRef<LayoutDimension> labels) const {
  SmallVector<int64_t> offset(labels.size(), 0);
  for (int i = 0; i < labels.size(); i++) {
    for (auto [name, it] : iterators) {
      if (name != labels[i])
        continue;
      offset[i] = it.getPosition();
    }
  }
  return offset;
}

// Get the offset into the SIMT vector corresponding to the incoming iterator.
// The offsets are projected onto the iterator. For example, if we have a vector
// mapping (batchx, batchy, vecx) and the iterator is (batchx, batchy), then
// we return an vector containing the offsets for (batchx, batchy).
SmallVector<int64_t> LayoutIterator::State::computeIteratorProjectedSIMTIndex(
    ArrayRef<LayoutDimension> labels) const {
  SmallVector<int64_t> indices = computeSIMTIndex(labels);
  SmallVector<int64_t> projectedIndices;
  for (int i = 0; i < labels.size(); i++) {
    for (auto [name, it] : iterators) {
      if (name == labels[i])
        projectedIndices.push_back(indices[i]);
    }
  }
  return projectedIndices;
}

int64_t LayoutAttr::getBatchDim(int64_t dim) {
  assert(dim < getLayouts().size());
  PerDimLayoutAttr layout = getDimLayout(dim);
  for (auto [name, shape] : llvm::zip(layout.getLabels(), layout.getShapes())) {
    if (isBatch(name.getValue()))
      return shape;
  }
  return -1;
}

static bool isIdentity(ArrayRef<int64_t> permutation) {
  for (int i = 0; i < permutation.size(); i++) {
    if (permutation[i] != i)
      return false;
  }
  return true;
}

static SmallVector<int64_t> getIdentityPermutation(int64_t size) {
  SmallVector<int64_t> permutation(size);
  for (int i = 0; i < size; i++) {
    permutation[i] = i;
  }
  return permutation;
}

// Check innermost vector dimension along cols to determine this value.
int64_t LayoutAttr::getTransferElements(ArrayRef<int64_t> permutation) const {
  if (!isIdentity(permutation)) {
    LayoutAttr permuted = llvm::cast<LayoutAttr>(permute(permutation));
    return permuted.getTransferElements(
        getIdentityPermutation(permutation.size()));
  }

  PerDimLayoutAttr colAttr = getDimLayout(getLayouts().size() - 1);
  return getInnermostVectorShape(colAttr.getLabels(), colAttr.getShapes());
}

std::optional<LayoutDimension> LayoutAttr::getLaneId(int64_t dim) const {
  auto layouts = getLayouts();
  assert(dim < layouts.size());
  PerDimLayoutAttr layout = layouts[dim];
  for (auto [nameAttr, shape] : llvm::zip(llvm::reverse(layout.getLabels()),
                                          llvm::reverse(layout.getShapes()))) {
    LayoutDimension name = nameAttr.getValue();
    if (isLane(name)) {
      return name;
    }
  }
  return std::nullopt;
}

SmallVector<int64_t>
LayoutAttr::projectSIMTVector(ArrayRef<LayoutDimension> labels,
                              ArrayRef<int64_t> input, int64_t dim) {
  SmallVector<int64_t> projectedVector;
  PerDimLayoutAttr layout = getDimLayout(dim);
  for (auto [name, value] : llvm::zip(labels, input)) {
    auto maybeShape = layout.getShape(name);
    if (!maybeShape) {
      projectedVector.push_back(value);
    }
  }
  return projectedVector;
}

// Layout conflict is one where
// 1. Layout A has a particular dim while Layout B doesn't and vice-versa
// 2. Layout A and B have a particular dim but their shapes disagree
bool LayoutAttr::hasLaneConflictWith(const LayoutAttr &other) {
  SmallVector<LayoutDimension> laneDims{
      LayoutDimension::LANEX, LayoutDimension::LANEY, LayoutDimension::LANEZ};

  if (getLayouts().size() != other.getLayouts().size())
    return true;

  for (auto [dimLayout, otherDimLayout] :
       llvm::zip_equal(getLayouts(), other.getLayouts())) {
    for (auto dim : laneDims) {
      auto shape0 = dimLayout.getShape(dim);
      auto shape1 = otherDimLayout.getShape(dim);
      if ((shape0 && !shape1) || (shape1 && !shape0))
        return true;
      if (shape0 && shape1)
        if (*shape0 != *shape1)
          return true;
    }
  }

  return false;
}

// Project out the layout for the specified dimensions
// resulting in the layout for a lower dimensional vector.
VectorLayoutInterface LayoutAttr::project(ArrayRef<bool> projectedDims) const {
  assert(projectedDims.size() == getLayouts().size() &&
         "projectedDims size must match layout size");

  ArrayRef<PerDimLayoutAttr> layouts = getLayouts();
  assert(projectedDims.size() == layouts.size());
  SmallVector<PerDimLayoutAttr> newLayouts;
  for (auto pair : llvm::zip(projectedDims, layouts)) {
    if (!std::get<0>(pair))
      newLayouts.push_back(std::get<1>(pair));
  }
  return LayoutAttr::get(getContext(), newLayouts);
}

// Permute the layout according to the provided permutation
// vector. The dimensionality of the layout remains the same.
VectorLayoutInterface LayoutAttr::permute(ArrayRef<int64_t> permutation) const {
  assert(permutation.size() == getLayouts().size() &&
         "permutation size must match layout size");

  ArrayRef<PerDimLayoutAttr> layouts = getLayouts();
  assert(permutation.size() == layouts.size());
  SmallVector<PerDimLayoutAttr> newLayouts;
  for (auto [index, permutedIndex] : llvm::enumerate(permutation)) {
    assert(permutedIndex >= 0 && permutedIndex < layouts.size());
    SmallVector<LayoutDimensionAttr> labels;
    SmallVector<int64_t> shapes;
    // Retain original batch dimension
    for (auto [label, shape] :
         llvm::zip(layouts[index].getLabels(), layouts[index].getShapes())) {
      if (isBatch(label.getValue())) {
        labels.push_back(label);
        shapes.push_back(shape);
        break;
      }
    }
    // Only permute the lane and vector dimensions
    for (auto [label, shape] : llvm::zip(layouts[permutedIndex].getLabels(),
                                         layouts[permutedIndex].getShapes())) {
      if (isBatch(label.getValue()))
        continue;
      labels.push_back(label);
      shapes.push_back(shape);
    }
    newLayouts.push_back(PerDimLayoutAttr::get(getContext(), labels, shapes));
  }
  return LayoutAttr::get(getContext(), newLayouts);
}
