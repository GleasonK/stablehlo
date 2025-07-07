/* Copyright 2025 The OpenXLA Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "third_party/stablehlo/google/builder/StablehloBuilder.h"

#include <cstdint>
#include <optional>

#include "third_party/llvm/llvm-project/llvm/include/llvm/Support/ErrorHandling.h"
#include "third_party/llvm/llvm-project/mlir/include/mlir/Dialect/Func/IR/FuncOps.h"
#include "third_party/llvm/llvm-project/mlir/include/mlir/IR/BuiltinAttributes.h"
#include "third_party/llvm/llvm-project/mlir/include/mlir/IR/BuiltinTypes.h"
#include "third_party/llvm/llvm-project/mlir/include/mlir/IR/Types.h"
#include "third_party/llvm/llvm-project/mlir/include/mlir/IR/Value.h"
#include "third_party/llvm/llvm-project/mlir/include/mlir/Interfaces/InferTypeOpInterface.h"
#include "third_party/llvm/llvm-project/mlir/include/mlir/Support/LLVM.h"
#include "third_party/stablehlo/google/builder/AttrTypeBuilderUtil.h"
#include "third_party/stablehlo/google/builder/MlirBuilder.h"
#include "third_party/stablehlo/stablehlo/dialect/StablehloOps.h"
#include "third_party/stablehlo/stablehlo/dialect/TypeInference.h"

namespace mlir {
namespace stablehlo {

/////////////////
// MANUAL APIs
/////////////////

bool isComplex(RankedTensorType type) {
  return mlir::isa<ComplexType>(type.getElementType());
}

MlirOp ConvertElementType(MlirOp input, Type resultElementType) {
  MlirOp operand = input;
  auto inputType = mlir::cast<RankedTensorType>(input.getType());
  auto resultType = inputType.clone(resultElementType);
  if (isa<ComplexType>(inputType.getElementType()) &&
      !isa<ComplexType>(resultElementType)) {
    operand = stablehlo::Real(operand);
  }
  return stablehlo::Convert(resultType, operand);
}

MlirOp ConvertElementType(MlirOp input, ElementType resultElementTypeKind) {
  auto resultElementType =
      getElementType(input.getContext(), resultElementTypeKind);
  return ConvertElementType(input, resultElementType);
}

// These are not finalized APIs, just an example of hiding the ugly for
// important leaf ops.
MlirOp Constant(MlirBuilder& builder, int64_t value) {
  return builder.create<stablehlo::ConstantOp>(DenseIntElementsAttr::get(
      RankedTensorType::get({}, builder.getOpBuilder().getI64Type()), value));
}
MlirOp Constant(MlirBuilder& builder, std::vector<int64_t> value) {
  auto numel = static_cast<int64_t>(value.size());
  return builder.create<stablehlo::ConstantOp>(DenseIntElementsAttr::get(
      RankedTensorType::get({numel}, builder.getOpBuilder().getI64Type()),
      value));
}

namespace {

// Use preferred element type, if not use LHS element type.
Type getDotResultType(RankedTensorType lhsType, ShapedTypeComponents type,
                      std::optional<ElementType> preferredResultType) {
  Type elementType =
      preferredResultType.has_value()
          ? getElementType(*lhsType.getContext(), preferredResultType.value())
          : lhsType.getElementType();
  return RankedTensorType::get(type.getDims(), elementType);
}

}  // namespace

MlirOp Dot(MlirOp lhs, MlirOp rhs, ArrayAttr precisionConfig,
           std::optional<ElementType> preferredElementType) {
  SmallVector<ShapedTypeComponents> inferredShape;
  auto lhsType = mlir::dyn_cast<RankedTensorType>(lhs.getType());
  auto rhsType = mlir::dyn_cast<RankedTensorType>(rhs.getType());
  if (!lhsType || !rhsType)
    llvm::report_fatal_error(
        "Failed to infer dot op type from lhs and rhs types.");

  if (failed(hlo::inferDotOp(lhs.getValue().getLoc(), lhsType, rhsType,
                             precisionConfig, inferredShape)) ||
      inferredShape.size() != 1)
    llvm::report_fatal_error(
        "Failed to infer dot op type from lhs and rhs types.");

  auto resultType =
      getDotResultType(lhsType, inferredShape[0], preferredElementType);
  return stablehlo::Dot(resultType, lhs, rhs, precisionConfig);
}

MlirOp DotGeneral(MlirOp lhs, MlirOp rhs, DotDimensionNumbersAttr dotDimsAttr,
                  ArrayAttr precisionConfig,
                  std::optional<ElementType> preferredElementType) {
  SmallVector<ShapedTypeComponents> inferredShape;
  auto lhsType = mlir::dyn_cast<RankedTensorType>(lhs.getType());
  auto rhsType = mlir::dyn_cast<RankedTensorType>(rhs.getType());
  if (!lhsType || !rhsType)
    llvm::report_fatal_error(
        "Failed to infer dot op type from lhs and rhs types.");

  if (failed(hlo::inferDotGeneralOp(lhs.getValue().getLoc(), lhsType, rhsType,
                                    dotDimsAttr.getLhsBatchingDimensions(),
                                    dotDimsAttr.getRhsBatchingDimensions(),
                                    dotDimsAttr.getLhsContractingDimensions(),
                                    dotDimsAttr.getRhsContractingDimensions(),
                                    precisionConfig, inferredShape)) ||
      inferredShape.size() != 1)
    llvm::report_fatal_error(
        "Failed to infer dot op type from lhs and rhs types.");

  auto resultType =
      getDotResultType(lhsType, inferredShape[0], preferredElementType);
  return stablehlo::DotGeneral(resultType, lhs, rhs, dotDimsAttr,
                               precisionConfig);
}

SmallVector<MlirOp> Arguments(RegionBuilder& rb, WhileOp op) {
  // Already init arguments, just return.
  if (!rb.getRegion().getArguments().empty()) {
    return wrap(rb, rb.getRegion().getArguments());
  }
  SmallVector<MlirOp> operands;
  operands.reserve(op.getOperands().size());
  for (auto operand : op.getOperands()) {
    operands.push_back(mlir::Argument(rb, operand.getType()));
  }
  return operands;
}

/////////////////
// GENERATED APIs
/////////////////

#include "third_party/stablehlo/google/builder/StablehloBuilder.cpp.inc"

}  // namespace stablehlo
}  // namespace mlir
