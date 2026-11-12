# SPDX-License-Identifier: BSD-3-Clause
# Run with bash. Build and freeze both sets of reference results.
set -euo pipefail

export RAYON_NUM_THREADS=2 CARGO_BUILD_JOBS=2 LC_ALL=C
repo=$(realpath "$(dirname "${BASH_SOURCE[0]}")/..")
compiler=$(g++ -dumpfullversion)
if [[ "$compiler" != 13.3.0 ]]; then
    echo 'Use g++ 13.3.0 to build the reference.' >&2
    exit 1
fi
build_dir=$(mktemp -d)
trap 'rm -rf "$build_dir"' EXIT

python3 "$repo/scripts/gen_corpus.py"
for level in O0 O2; do
    g++ -std=c++11 "-$level" -I"$repo/vendor/liblinear" \
        "$repo/reference/driver/ref_driver.cpp" \
        "$repo/vendor/liblinear/linear.cpp" \
        "$repo/vendor/liblinear/newton.cpp" \
        "$repo/vendor/liblinear/blas/daxpy.c" \
        "$repo/vendor/liblinear/blas/ddot.c" \
        "$repo/vendor/liblinear/blas/dnrm2.c" \
        "$repo/vendor/liblinear/blas/dscal.c" \
        -o "$build_dir/ref_driver_$level"
done

{
    echo '# Reference build'
    echo
    echo 'Source: LIBLINEAR 2.50, tag `v250`.'
    echo 'Commit: `491c9f1188b97ba70847c70a68be363d186ddf9d`.'
    echo
    echo 'Compiler:'
    echo '```text'
    g++ --version | head -n 1
    echo '```'
    echo
    echo "Machine: $(uname -srm)."
    echo "C library: $(getconf GNU_LIBC_VERSION)."
    echo "Python: $(python3 --version)."
    echo
    echo 'Each command runs from the repo root. Output files use a temporary folder.'
    echo 'All source files, including the four BLAS files, use the same flags.'
    echo '```sh'
    for level in O0 O2; do
        echo "g++ -std=c++11 -$level -Ivendor/liblinear reference/driver/ref_driver.cpp vendor/liblinear/linear.cpp vendor/liblinear/newton.cpp vendor/liblinear/blas/daxpy.c vendor/liblinear/blas/ddot.c vendor/liblinear/blas/dnrm2.c vendor/liblinear/blas/dscal.c -o <temporary>/ref_driver_$level"
    done
    echo '```'
    echo
    echo 'No other compiler flags are used.'
    echo 'The driver calls `srand(1)` before each `train` call.'
    echo 'The driver sets the C locale and prints each double with `%.17g`.'
    echo 'The runs use `RAYON_NUM_THREADS=2`, `CARGO_BUILD_JOBS=2` and `LC_ALL=C`.'
    echo 'Builds and case runs are serial.'
    echo
    echo 'Run `bash scripts/freeze_reference.sh` to build and write both result sets.'
    echo 'Run `python3 scripts/check_reference.py` to check the saved files and rerun all cases twice.'
} > "$repo/reference/BUILD.md"

python3 "$repo/scripts/check_reference.py" --record \
    --driver-o0 "$build_dir/ref_driver_O0" --driver-o2 "$build_dir/ref_driver_O2"
python3 "$repo/scripts/check_reference.py" --driver-o2 "$build_dir/ref_driver_O2"
