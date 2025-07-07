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

#ifndef STABLEHLO_BUILDER_CHLOBUILDER_H_
#define STABLEHLO_BUILDER_CHLOBUILDER_H_

#include <cstdint>

#include "third_party/llvm/llvm-project/llvm/include/llvm/ADT/SmallVector.h"
#include "third_party/stablehlo/google/builder/MlirBuilder.h"
#include "third_party/stablehlo/stablehlo/dialect/ChloOps.h"

namespace mlir {
namespace chlo {

/////////////////
// GENERATED APIs
/////////////////

#include "third_party/stablehlo/google/builder/ChloBuilder.h.inc"

}  // namespace chlo
}  // namespace mlir

#endif  // STABLEHLO_BUILDER_CHLOBUILDER_H_
