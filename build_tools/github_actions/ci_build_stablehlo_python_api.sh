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

if [[ $# -ne 2 ]] ; then
  echo "Usage: $0 <llvm_project_dir> <stablehlo_api_build_dir>>"
  exit 1
fi

LLVM_PROJECT_DIR="$1"
STABLEHLO_PYTHON_BUILD_DIR="$2"

# Configure StableHLO Python Bindings
BUILD_TOOLS_DIR="$(dirname "$(readlink -f "$0")")"
$BUILD_TOOLS_DIR/ci_build_llvm.sh "$1" "$2"


# Build and Check StableHLO Python Bindings
cd "$STABLEHLO_PYTHON_BUILD_DIR"
ninja check-stablehlo-python
