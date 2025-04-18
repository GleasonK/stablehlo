# Copyright 2024 The StableHLO Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name StablehloLinalgTransforms)
add_public_tablegen_target(StablehloLinalgTransformsPassIncGen)

add_mlir_library(StablehloLinalgTransforms
  LegalizeToLinalgUtils.cpp
  StablehloLegalizeToLinalg.cpp
  StablehloToArith.cpp
  StablehloToLinalgConvolution.cpp
  StablehloToLinalgDotProduct.cpp
  StablehloToLinalgPointwise.cpp
  StablehloToLinalgRandom.cpp
  StablehloToLinalgReduce.cpp
  TypeConversion.cpp

  DEPENDS
  StablehloOpsIncGen
  StablehloLinalgTransformsPassIncGen

  LINK_LIBS PUBLIC
  ChloOps
  StablehloBase
  StablehloOps
  StablehloTypeConversion
  MLIRArithDialect
  MLIRBufferizationDialect
  MLIRComplexDialect
  MLIRFuncDialect
  MLIRIR
  MLIRLinalgDialect
  MLIRLinalgTransforms
  MLIRLinalgUtils
  MLIRMathDialect
  MLIRMemRefDialect
  MLIRPass
  MLIRSCFDialect
  MLIRShapeDialect
  MLIRSparseTensorDialect
  MLIRSupport
  MLIRTensorDialect
  MLIRTransforms
  MLIRVectorDialect
)
