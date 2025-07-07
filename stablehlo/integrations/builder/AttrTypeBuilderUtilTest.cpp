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

#include <cstdint>
#include <string>
#include <utility>

#include "testing/base/public/gunit.h"
#include "third_party/llvm/llvm-project/llvm/include/llvm/ADT/DenseMap.h"
#include "third_party/llvm/llvm-project/mlir/include/mlir/IR/BuiltinTypes.h"
#include "third_party/llvm/llvm-project/mlir/include/mlir/IR/Location.h"
#include "third_party/llvm/llvm-project/mlir/include/mlir/IR/MLIRContext.h"
#include "third_party/llvm/llvm-project/mlir/include/mlir/IR/Types.h"
#include "third_party/llvm/llvm-project/mlir/include/mlir/Support/DebugStringHelper.h"
#include "third_party/llvm/llvm-project/mlir/include/mlir/Support/LLVM.h"
#include "third_party/stablehlo/google/builder/AttrTypeBuilderUtil.h"

namespace mlir {

TEST(AttrTypeBuilderUtilTest, TestMakeTensorType) {
  MLIRContext context;
  llvm::DenseMap<std::pair<SmallVector<int64_t>, ElementType>, std::string>
      testCaseMap = {
          {{{}, ElementType::PRED}, "tensor<i1>"},
          {{{}, ElementType::I8}, "tensor<i8>"},
          {{{}, ElementType::I16}, "tensor<i16>"},
          {{{}, ElementType::I32}, "tensor<i32>"},
          {{{}, ElementType::I64}, "tensor<i64>"},
          {{{}, ElementType::UI8}, "tensor<ui8>"},
          {{{}, ElementType::UI16}, "tensor<ui16>"},
          {{{}, ElementType::UI32}, "tensor<ui32>"},
          {{{}, ElementType::UI64}, "tensor<ui64>"},
          {{{}, ElementType::BF16}, "tensor<bf16>"},
          {{{}, ElementType::F16}, "tensor<f16>"},
          {{{}, ElementType::F32}, "tensor<f32>"},
          {{{}, ElementType::F64}, "tensor<f64>"},
          {{{}, ElementType::F4E2M1FN}, "tensor<f4E2M1FN>"},
          {{{}, ElementType::F6E2M3FN}, "tensor<f6E2M3FN>"},
          {{{}, ElementType::F6E3M2FN}, "tensor<f6E3M2FN>"},
          {{{}, ElementType::F8E3M4}, "tensor<f8E3M4>"},
          {{{}, ElementType::F8E4M3}, "tensor<f8E4M3>"},
          {{{}, ElementType::F8E4M3FN}, "tensor<f8E4M3FN>"},
          {{{}, ElementType::F8E4M3FNUZ}, "tensor<f8E4M3FNUZ>"},
          {{{}, ElementType::F8E4M3B11FNUZ}, "tensor<f8E4M3B11FNUZ>"},
          {{{}, ElementType::F8E5M2}, "tensor<f8E5M2>"},
          {{{}, ElementType::F8E5M2FNUZ}, "tensor<f8E5M2FNUZ>"},
          {{{}, ElementType::F8E8M0FNU}, "tensor<f8E8M0FNU>"},
          {{{1}, ElementType::F64}, "tensor<1xf64>"},
          {{{1, 2, 3}, ElementType::F64}, "tensor<1x2x3xf64>"},
      };
  for (auto& [inputs, value] : testCaseMap) {
    RankedTensorType type =
        makeTensorType(context, inputs.first, inputs.second);
    EXPECT_EQ(value, debugString(type));
  }
}

}  // namespace mlir
