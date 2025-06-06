# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
# Copyright 2023 The StableHLO Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
include(AddMLIRPython)

################################################################################
# Sources
################################################################################

# Note that the directory structure for source files is meaningful. For example,
# putting .td and .py files under . instead of mlir/python will break things,
# even if the build rules below are adjusted accordingly.

declare_mlir_python_sources(CheckPythonSources)
declare_mlir_python_sources(CheckPythonSources.Dialects
  ADD_TO_PARENT CheckPythonSources
)

declare_mlir_dialect_python_bindings(
  ADD_TO_PARENT CheckPythonSources.Dialects
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/mlir"
  TD_FILE dialects/CheckOps.td
  SOURCES dialects/check.py
  DIALECT_NAME check)

declare_mlir_python_sources(ChloPythonSources)
declare_mlir_python_sources(ChloPythonSources.Dialects
  ADD_TO_PARENT ChloPythonSources
)

declare_mlir_dialect_python_bindings(
  ADD_TO_PARENT ChloPythonSources.Dialects
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/mlir"
  TD_FILE dialects/ChloOps.td
  SOURCES dialects/chlo.py
  DIALECT_NAME chlo)

declare_mlir_python_sources(StablehloPythonSources)
declare_mlir_python_sources(StablehloPythonSources.Dialects
  ADD_TO_PARENT StablehloPythonSources
)

declare_mlir_dialect_python_bindings(
  ADD_TO_PARENT StablehloPythonSources.Dialects
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/mlir"
  TD_FILE dialects/StablehloOps.td
  SOURCES dialects/stablehlo.py
  DIALECT_NAME stablehlo)

declare_mlir_python_sources(StablehloToSavedModelPythonSources
  ADD_TO_PARENT StablehloPythonSources
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}"
  SOURCES
    stablehlo/savedmodel/stablehlo_to_tf_saved_model.py
)

declare_mlir_python_sources(StablehloTestdataGeneratorPythonSources
  ADD_TO_PARENT StablehloPythonSources
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}"
  SOURCES
    stablehlo/testdata_generator/testdata_execution_utils.py
    stablehlo/testdata_generator/testdata_generator_lib.py
    stablehlo/testdata_generator/testdata_processor.py
)

declare_mlir_python_sources(VhloPythonSources)
declare_mlir_python_sources(VhloPythonSources.Dialects
  ADD_TO_PARENT VhloPythonSources
)

declare_mlir_dialect_python_bindings(
  ADD_TO_PARENT VhloPythonSources.Dialects
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/mlir"
  TD_FILE dialects/VhloOps.td
  SOURCES dialects/vhlo.py
  DIALECT_NAME vhlo)

################################################################################
# Extensions
################################################################################

## The following options are used specifically for compiling nanobind codebase.
add_compile_options(-Wno-error=cast-qual -Wno-error=nested-anon-types
  -Wno-error=extra-semi -Wno-error=zero-length-array
  -Wno-error=covered-switch-default)

declare_mlir_python_sources(CheckPythonExtensions)
declare_mlir_python_extension(CheckPythonExtensions.Main
  MODULE_NAME _check
  ADD_TO_PARENT CheckPythonExtensions
  PYTHON_BINDINGS_LIBRARY nanobind
  SOURCES
  CheckModule.cpp
  EMBED_CAPI_LINK_LIBS
  CheckCAPI
  PRIVATE_LINK_LIBS
    LLVMSupport
)

declare_mlir_python_sources(ChloPythonExtensions)
declare_mlir_python_extension(ChloPythonExtensions.Main
  MODULE_NAME _chlo
  ADD_TO_PARENT ChloPythonExtensions
  PYTHON_BINDINGS_LIBRARY nanobind
  SOURCES
    ChloModule.cpp
  EMBED_CAPI_LINK_LIBS
    ChloCAPI
  PRIVATE_LINK_LIBS
    LLVMSupport
)

declare_mlir_python_sources(StablehloPythonExtensions)
declare_mlir_python_extension(StablehloPythonExtensions.Main
  MODULE_NAME _stablehlo
  ADD_TO_PARENT StablehloPythonExtensions
  PYTHON_BINDINGS_LIBRARY nanobind
  SOURCES
    StablehloApi.cpp
    StablehloModule.cpp
  EMBED_CAPI_LINK_LIBS
    StablehloCAPI
  PRIVATE_LINK_LIBS
    StablehloCAPI
    LLVMSupport
)

declare_mlir_python_sources(VhloPythonExtensions)
declare_mlir_python_extension(VhloPythonExtensions.Main
  MODULE_NAME _vhlo
  ADD_TO_PARENT VhloPythonExtensions
  PYTHON_BINDINGS_LIBRARY nanobind
  SOURCES
    VhloModule.cpp
  EMBED_CAPI_LINK_LIBS
    VhloCAPI
  PRIVATE_LINK_LIBS
    LLVMSupport
)

################################################################################
# Generate packages and shared libraries
################################################################################

# Note that we are deliberately putting CHLO and StableHLO dialects together
# into a single package, even though they are independent from each other.
# That's because the initialization code generated for these packages modifies
# global state, so importing multiple packages like that will mess things up.

add_mlir_python_common_capi_library(StablehloUnifiedPythonCAPI
  INSTALL_COMPONENT StablehloUnifiedPythonModules
  INSTALL_DESTINATION python_packages/stablehlo/mlir/_mlir_libs
  OUTPUT_DIRECTORY "${STABLEHLO_BINARY_DIR}/python_packages/stablehlo/mlir/_mlir_libs"
  RELATIVE_INSTALL_ROOT "../../../.."
  DECLARED_SOURCES
    MLIRPythonSources
    MLIRPythonExtension.RegisterEverything
    CheckPythonSources
    CheckPythonExtensions
    ChloPythonSources
    ChloPythonExtensions
    StablehloPythonSources
    StablehloPythonExtensions
    VhloPythonSources
    VhloPythonExtensions
)

add_mlir_python_modules(StablehloUnifiedPythonModules
  ROOT_PREFIX "${STABLEHLO_BINARY_DIR}/python_packages/stablehlo/mlir"
  INSTALL_PREFIX "python_packages/stablehlo/mlir"
  DECLARED_SOURCES
    MLIRPythonSources
    MLIRPythonExtension.RegisterEverything
    CheckPythonSources
    CheckPythonExtensions
    ChloPythonSources
    ChloPythonExtensions
    StablehloPythonSources
    StablehloPythonExtensions
    VhloPythonSources
    VhloPythonExtensions
  COMMON_CAPI_LINK_LIBS
    StablehloUnifiedPythonCAPI
)

################################################################################
# Tests
################################################################################

add_subdirectory(tests)
