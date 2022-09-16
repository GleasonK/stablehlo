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

print_usage() {
  echo "Usage: $0 [-f] <llvm_project_dir> <llvm_build_dir> <stablehlo_build_dir>"
  echo "    -f   Auto-fix clang-tidy issues."
}

FIX_FLAG=''
while getopts 'f' flag; do
  case "${flag}" in
    f) ='-fix' ;;
    *) print_usage
       exit 1 ;;
  esac
done
shift $(( OPTIND - 1 ))

if [[ $# -ne 3 ]] ; then
  print_usage
  exit 1
fi

LLVM_PROJECT_DIR="$1"
LLVM_BUILD_DIR="$2"
STABLEHLO_BUILD_DIR="$3"

# Setup. Do not build StableHLO
BUILD_TOOLS_DIR="$(dirname "$(readlink -f "$0")")"

set -x
echo "Configuring and building LLVM..."
$BUILD_TOOLS_DIR/ci_build_llvm.sh "$1" "$2" > /dev/null

echo "Configuring StableHLO..."
$BUILD_TOOLS_DIR/ci_build_stablehlo.sh -n "$LLVM_PROJECT_DIR" "$STABLEHLO_BUILD_DIR" > /dev/null

# Exclude python files since the current build is only for source files.
echo "Running clang-tidy..."
BASE_BRANCH=${GITHUB_BASE_REF:-origin}
CLANG_TIDY_FILES=$(git diff --name-only HEAD $BASE_BRANCH | \
                   grep '.*\.h\|.*\.cpp' | \
                   grep -v "/python/" | \
                   xargs)

clang-tidy $FIX_FLAG -p $STABLEHLO_BUILD_DIR -warnings-as-errors='*' $CLANG_TIDY_FILES
echo $?
set +x