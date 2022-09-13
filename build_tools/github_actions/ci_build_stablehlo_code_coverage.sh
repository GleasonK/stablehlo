#!/bin/bash
# Copyright 2022 The StableHLO Authors.
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

# This file is similar to build_mlir.sh, but passes different flags for
# caching in GitHub Actions.

# This file gets called on build directory where resources are placed
# during `ci_configure`, and builds stablehlo in the directory specified
# by the second argument.

if [[ $# -ne 2 ]] ; then
  echo "Usage: $0 <llvm_build_dir> <stablehlo_build_dir>"
  exit 1
fi

LLVM_BUILD_DIR="$1"
STABLEHLO_BUILD_DIR="$2"

# Configure StableHLO
cmake -GNinja \
  -B"$STABLEHLO_BUILD_DIR" \
  -DLLVM_ENABLE_LLD=ON \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_ENABLE_ASSERTIONS=On \
  -DMLIR_DIR="$LLVM_BUILD_DIR/lib/cmake/mlir" \
  -DCMAKE_CXX_COMPILER=clang++ \
  -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
  -DCMAKE_CXX_FLAGS="${CMAKE_CXX_FLAGS} -fprofile-instr-generate -fprofile-arcs -ftest-coverage -fcoverage-mapping" \
  -DCMAKE_C_COMPILER=clang \
  -DCMAKE_C_COMPILER_LAUNCHER=ccache \
  -DCMAKE_C_FLAGS="${CMAKE_C_FLAGS} -fprofile-instr-generate -fprofile-arcs -ftest-coverage -fcoverage-mapping"

# Build and Check StableHLO
(cd "$STABLEHLO_BUILD_DIR" && ninja check-stablehlo)

BUILD_TOOLS_DIR="$(dirname "$(readlink -f "$0")")"

REPORT_DATE="$(date +'%Y_%m_%d_%H-%M-%S')"
REPORT_DIR="reports/ccov_$REPORT_DATE"

mkdir $REPORT_DIR

lcov --directory "$STABLEHLO_BUILD_DIR"  \
     --base-directory "stablehlo" \
     --gcov-tool "$BUILD_TOOLS_DIR/llvm-gcov.sh" \
     --capture -o /tmp/cov.info \
     --no-external \
     --exclude "*.inc"
     --exclude "llvm-project/*"

# Capture code coverage output
exec 5>&1
CCOV_OUT=$(genhtml /tmp/cov.info \
    --output-directory $REPORT_DIR \
    --ignore-errors source \
    --sort \
    --prefix $(pwd) | tee /dev/fd/5) 

# Make coverage description for readme. Parses the following output:
# Overall coverage rate:
#  lines......: 43.2% (2463 of 5702 lines)
#  functions..: 36.7% (352 of 958 functions)
# ====> Line coverage: 43.2% [2463/5702], Function coverage: 36.7% [352/958]
CCOV_DESC=$(echo "${CCOV_OUT}" | grep "Overall coverage" -A2 | grep -o '[0-9]\+\.\?[0-9]\+\?%\?' | \
            xargs printf "Line coverage: %s [%d/%d], Function coverage: %s [%d/%d]") 

echo "Adding readme entry for:"
echo "  $CCOV_DESC"
echo " - [$REPORT_DATE](https://htmlpreview.github.io/?https://github.com/GleasonK/stablehlo/blob/ccov/$REPORT_DIR/index.html) $CCOV_DESC" >> reports/README.md
