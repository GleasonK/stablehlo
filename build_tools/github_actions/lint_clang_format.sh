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
  echo "Usage: $0 [-f]"
  echo "    -f   Auto-fix clang-format issues."
}

FORMAT_MODE='validate'
while getopts 'f' flag; do
  case "${flag}" in
    f) FORMAT_MODE='fix' ;;
    *) print_usage
       exit 1 ;;
  esac
done
shift $(( OPTIND - 1 ))

if [[ $# -ne 0 ]] ; then
  print_usage
  exit 1
fi

echo "Gathering changed files..."
BASE_BRANCH=${GITHUB_BASE_REF:-origin}
CLANG_FILES=$(git diff --name-only HEAD $BASE_BRANCH | grep '.*\.h\|.*\.cpp' | xargs)
echo "  Files: $CLANG_FILES"

echo "Running clang-format [mode=$FORMAT_MODE]..."
if [[ $DO_FIX == 'true' ]]; then
  clang-format --style=google -i $CLANG_FILES
else
  clang-format --style=google --dry-run --Werror $CLANG_FILES
fi