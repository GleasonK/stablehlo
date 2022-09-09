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

if [[ $# -ne 3 ]] ; then
  echo "Usage: $0 <llvm_project_dir> <llvm_build_dir> <stablehlo_build_dir>"
  exit 1
fi

LLVM_PROJECT_DIR="$1"
LLVM_BUILD_DIR="$2"
STABLEHLO_BUILD_DIR="$2"

# Setup. Do not build StableHLO
BUILD_TOOLS_DIR="$(dirname "$(readlink -f "$0")")"

echo "Configuring and building LLVM..."
$BUILD_TOOLS_DIR/ci_build_llvm.sh "$1" "$2" > /dev/null

echo "Configuring StableHLO..."
$BUILD_TOOLS_DIR/ci_build_stablehlo.sh -n "$2" "$3" > /dev/null

echo "Running clang-tidy..."
CLANG_TIDY="$LLVM_PROJECT_DIR/clang-tools-extra/clang-tidy/tool/clang-tidy-diff.py"
git diff origin/main HEAD | $CLANG_TIDY -p1 -path $STABLEHLO_BUILD_DIR
echo "${PIPESTATUS[0]} ${PIPESTATUS[1]}"