# C reference tables

Set `REPO` to the absolute path of this repository. These commands use g++
13.3.0 and the unchanged C BLAS source. They need no network. Each build runs
alone. The table holds f64 bits as 16-digit hex strings, so that JSON number
rounding cannot change a value. NaN checks require the NaN class; IEEE
arithmetic does not fix a NaN sign or payload. All other values use exact bits.

```sh
RAYON_NUM_THREADS=2 CARGO_BUILD_JOBS=2 g++ -std=c++11 -O2 -I "$REPO/vendor/liblinear/blas" "$REPO/tests/fixtures/blas_table.c" "$REPO/vendor/liblinear/blas/daxpy.c" "$REPO/vendor/liblinear/blas/ddot.c" "$REPO/vendor/liblinear/blas/dnrm2.c" "$REPO/vendor/liblinear/blas/dscal.c" -o /tmp/liblinear_blas_table
/tmp/liblinear_blas_table > "$REPO/tests/fixtures/blas_table.json"
sha256sum "$REPO/tests/fixtures/blas_table.json"
```

The output hash is stored in `blas_table.sha256` next to the table. The 11
entries include lengths 0, 1, 4, 5, 6 and 9, large values, subnormal values,
signed zeros, a zero scale with nonfinite inputs and a nonfinite vector.

The two extra solver cases use the frozen reference driver. The multiclass
case has free bias, repeated weights for label 7 and a weight for absent label
99. The absent label is ignored, as in upstream. The binary case starts with
label -1 and has zero bias, so it checks the swap to put label +1 first.
They are separate from the 15 frozen cases and do not change those files.

```sh
RAYON_NUM_THREADS=2 CARGO_BUILD_JOBS=2 g++ -std=c++11 -O2 -I "$REPO/vendor/liblinear" "$REPO/reference/driver/ref_driver.cpp" "$REPO/vendor/liblinear/linear.cpp" "$REPO/vendor/liblinear/newton.cpp" "$REPO/vendor/liblinear/blas/daxpy.c" "$REPO/vendor/liblinear/blas/ddot.c" "$REPO/vendor/liblinear/blas/dnrm2.c" "$REPO/vendor/liblinear/blas/dscal.c" -o /tmp/liblinear_solver2_reference
/tmp/liblinear_solver2_reference "$REPO/tests/fixtures/multiclass.txt" "$REPO/tests/fixtures/multiclass.json"
/tmp/liblinear_solver2_reference "$REPO/tests/fixtures/binary_labels.txt" "$REPO/tests/fixtures/binary_labels.json"
```

## C random table

`rand_table.c` prints the first 10000 values after each of `srand(1)` and
`srand(2)`. The build uses g++ 13.3.0 and glibc 2.39. The output hash is in
`rand_table.sha256`. Set `REPO` to the absolute repository path.

```sh
RAYON_NUM_THREADS=2 CARGO_BUILD_JOBS=2 g++ -std=c++11 -O2 "$REPO/tests/fixtures/rand_table.c" -o /tmp/liblinear_rand_table
/tmp/liblinear_rand_table > "$REPO/tests/fixtures/rand_table.json"
sha256sum "$REPO/tests/fixtures/rand_table.json"
```

## Extra dual cases

`dual_reference.cpp` uses the frozen reader, writer and training code.
The last argument sets `w_recalc`. The multiclass case checks one stream
across three class fits, repeated class weights and an absent weight label.
The binary case checks label order and zero bias. Both check the weight
rebuild branch with C results. The driver also counts Newton fallbacks.
The frozen `analytic_l2svc` case gives one fallback; the extra cases give zero.

```sh
RAYON_NUM_THREADS=2 CARGO_BUILD_JOBS=2 g++ -std=c++11 -O2 -I "$REPO/vendor/liblinear" "$REPO/tests/fixtures/dual_reference.cpp" "$REPO/vendor/liblinear/linear.cpp" "$REPO/vendor/liblinear/newton.cpp" "$REPO/vendor/liblinear/blas/daxpy.c" "$REPO/vendor/liblinear/blas/ddot.c" "$REPO/vendor/liblinear/blas/dnrm2.c" "$REPO/vendor/liblinear/blas/dscal.c" -o /tmp/liblinear_dual_reference
/tmp/liblinear_dual_reference "$REPO/tests/fixtures/dual_multiclass.txt" "$REPO/tests/fixtures/dual_multiclass.json" 0
/tmp/liblinear_dual_reference "$REPO/tests/fixtures/dual_multiclass.txt" "$REPO/tests/fixtures/dual_multiclass_recalc.json" 1
/tmp/liblinear_dual_reference "$REPO/tests/fixtures/dual_binary_labels.txt" "$REPO/tests/fixtures/dual_binary_labels.json" 0
/tmp/liblinear_dual_reference "$REPO/tests/fixtures/dual_binary_labels.txt" "$REPO/tests/fixtures/dual_binary_labels_recalc.json" 1
/tmp/liblinear_dual_reference "$REPO/reference/corpus/analytic_l2svc.txt" /tmp/liblinear_dual_analytic.json 0
```

## C libm table

`libm_table.c` uses a fixed seed of 1 to print 2000 exp inputs and outputs
and 2000 log inputs and outputs with `%.17g`. The exp inputs include
1e-300 through 700, -700 through zero, and signed zero. The log inputs
span 1e-300 through 1e300 and include values near 1. The build uses g++
13.3.0 with `-O2` and glibc 2.39. `libm_table.json` stores the build command,
compiler, C library and SHA-256 of the raw output. Keep the printed numbers
when adding metadata, so signed zero is preserved. Set `REPO` to the
absolute repository path.

```sh
RAYON_NUM_THREADS=2 CARGO_BUILD_JOBS=2 g++ -std=c++11 -O2 "$REPO/tests/fixtures/libm_table.c" -o /tmp/liblinear_libm_table
/tmp/liblinear_libm_table > /tmp/liblinear_libm_table_output.json
sha256sum /tmp/liblinear_libm_table_output.json
```

The Rust test compares each output by bits and prints the runner's C
library version. A different bit fails the table test.

## Extra logistic cases

`lr_reference.cpp` uses the frozen reader, writer and training code.
The last argument sets initial weights to `i / 100.0` when it is 1.
The multiclass case checks three class fits, repeated class weights and an
absent weight label. The binary case checks label order and zero bias.
The free-bias case checks both default and initial weights. The fallback
case uses C of 1e6 and eps of 1e-16; the driver counts two switches to
primal logistic Newton. These cases do not change the frozen corpus.

```sh
RAYON_NUM_THREADS=2 CARGO_BUILD_JOBS=2 g++ -std=c++11 -O2 -I "$REPO/vendor/liblinear" "$REPO/tests/fixtures/lr_reference.cpp" "$REPO/vendor/liblinear/linear.cpp" "$REPO/vendor/liblinear/newton.cpp" "$REPO/vendor/liblinear/blas/daxpy.c" "$REPO/vendor/liblinear/blas/ddot.c" "$REPO/vendor/liblinear/blas/dnrm2.c" "$REPO/vendor/liblinear/blas/dscal.c" -o /tmp/liblinear_lr_reference
/tmp/liblinear_lr_reference "$REPO/tests/fixtures/lr_multiclass.txt" "$REPO/tests/fixtures/lr_multiclass.json" 0
/tmp/liblinear_lr_reference "$REPO/tests/fixtures/lr_binary_labels.txt" "$REPO/tests/fixtures/lr_binary_labels.json" 0
/tmp/liblinear_lr_reference "$REPO/tests/fixtures/lr_free_bias.txt" "$REPO/tests/fixtures/lr_free_bias.json" 0
/tmp/liblinear_lr_reference "$REPO/tests/fixtures/lr_free_bias.txt" "$REPO/tests/fixtures/lr_free_bias_initial.json" 1
/tmp/liblinear_lr_reference "$REPO/tests/fixtures/lr_fallback.txt" "$REPO/tests/fixtures/lr_fallback.json" 0
```

## Extra L1 problems

`l1_reference.cpp` uses the frozen reader, writer and vendored training code.
It takes a case path and an output path. Each `l1_*.txt` case selects solvers
5 and 6. `l1_extra.json` stores all six results. Its input hashes keep each
result tied to its case. `l1_extra.build.json` stores the build commands,
compiler, C library, source hashes, result hash and each raw output hash.
The O0 and O2 raw outputs match by bytes for each extra problem.
Set `REPO` to the absolute repository path. Create `/tmp/liblinear-l1` first.

```sh
RAYON_NUM_THREADS=2 CARGO_BUILD_JOBS=2 g++ -std=c++11 -O0 --coverage -I "$REPO/vendor/liblinear" "$REPO/tests/fixtures/l1_reference.cpp" "$REPO/vendor/liblinear/linear.cpp" "$REPO/vendor/liblinear/newton.cpp" "$REPO/vendor/liblinear/blas/daxpy.c" "$REPO/vendor/liblinear/blas/ddot.c" "$REPO/vendor/liblinear/blas/dnrm2.c" "$REPO/vendor/liblinear/blas/dscal.c" -o /tmp/liblinear-l1/l1_reference_O0
RAYON_NUM_THREADS=2 CARGO_BUILD_JOBS=2 g++ -std=c++11 -O2 -I "$REPO/vendor/liblinear" "$REPO/tests/fixtures/l1_reference.cpp" "$REPO/vendor/liblinear/linear.cpp" "$REPO/vendor/liblinear/newton.cpp" "$REPO/vendor/liblinear/blas/daxpy.c" "$REPO/vendor/liblinear/blas/ddot.c" "$REPO/vendor/liblinear/blas/dnrm2.c" "$REPO/vendor/liblinear/blas/dscal.c" -o /tmp/liblinear-l1/l1_reference_O2
for name in tiny zero_columns one_class duplicate C_small C_large; do
    LC_ALL=C /tmp/liblinear-l1/l1_reference_O0 "$REPO/tests/fixtures/l1_$name.txt" "/tmp/liblinear-l1/${name}_O0.json"
    LC_ALL=C /tmp/liblinear-l1/l1_reference_O2 "$REPO/tests/fixtures/l1_$name.txt" "/tmp/liblinear-l1/${name}_O2.json"
    cmp "/tmp/liblinear-l1/${name}_O0.json" "/tmp/liblinear-l1/${name}_O2.json"
    sha256sum "/tmp/liblinear-l1/${name}_O2.json"
done
```

The aggregate stores a `problems` array. Each entry has `name`,
`input_sha256` and the raw C `runs` array. Preserve the C number strings when
assembling this array. The Rust test checks the aggregate and source hashes
before it compares every model field by bits.

Coverage uses `gcov -j -b /tmp/liblinear-l1/l1_reference_O0-linear.gcno`.
Run gcov with `/tmp/liblinear-l1` as its working folder. Remove the `.gcda`
files in that folder before each separate coverage run. Run all 28 solver
5 and 6 corpus cases to get the combined corpus counts. Run each extra case
alone to get its counts. The saved coverage rows use source line numbers,
execution counts and branch edge counts in gcov order. The source hash ties
these lines to the frozen `linear.cpp`.

The large-C case reaches solver 5's 20-step line-search exit three times.
The one-class case reaches solver 6's exit six times. The corpus reaches
neither exit. The contract lists the shrinking paths and their measured
counts. These inputs reach step limits; no convergence or speed claim is made.


## Solver 4 multiclass and sort tables

`mcsvm_reference.cpp` uses the frozen reader and writer and the unchanged
vendored solver. It prints doubles with `%.17g` and starts seed 1. The six
`mcsvm_*.txt` inputs and their C results are in `mcsvm_extra.json`.
`mcsvm_extra.build.json` stores compiler, commands, source and output
hashes, and GCC coverage counts. All six O0 and O2 outputs are equal by
bytes. Coverage uses the O0 program and the unchanged `linear.cpp`.

Run the code below from any folder. Replace `{repo}` with the absolute
repository path. This command rebuilds both C++ programs, makes the six
inputs again with Python seed 410, checks both outputs, and saves the
extra results and coverage record. It does not write to the frozen corpus.

```sh
RAYON_NUM_THREADS=2 CARGO_BUILD_JOBS=2 python3 {repo}/tests/fixtures/mcsvm_reference.py
```

`mcsvm_sort.c` uses seed 410 with a 32-bit linear generator. Its multiplier
is 1664525 and its increment is 1013904223. Each draw uses the upper 16
bits. Arrays have 1 through 16 entries drawn from the listed finite-double
palette. The output stores sorted palette indices as hex digits. Thus it
keeps all bits, including the zero sign. The Rust unit test makes the same
inputs and checks each sorted value by bits. The table has 10000 arrays.
The build and run commands and SHA-256 hashes are in
`mcsvm_sort.build.json`. The sort uses the same comparison as the source.

The normal solver 4 tests check every corpus and extra output field against
`reference/tolerances.json`. They also check the fixture input, source and
output hashes. `reference/parity/solver4.json` is made by the corpus test.
The contract gives all branch counts and the expected sort-mutation error.


## Extra support vector regression problems

`svr_reference.cpp` uses the frozen reader and writer and the unchanged
vendored solvers 11, 12 and 13. It takes a case path, output path and an
optional solver number for separate branch counts. All doubles use `%.17g`.
Each train call starts seed 1. The six `svr_*.txt` inputs have 12 rows,
five features and bias 1. They cover equal targets, one outlier, p zero,
p 100, a copied row with C 1e-3, and a zero column with C 1e3 and eps 1e-30.

Replace `{repo}` with the absolute repository path. Run this command from
any folder. It builds O0 coverage and O2 programs in `target/svr-reference`,
makes the inputs with Python seed 1113, checks byte equality between both
C++ builds, and saves `svr_extra.json` and `svr_extra.build.json`. The script
needs no network. It runs each build and program alone. It changes no
frozen corpus, golden file or tolerance.

```sh
RAYON_NUM_THREADS=2 CARGO_BUILD_JOBS=2 python3 {repo}/tests/fixtures/svr_reference.py
```

The build record contains both build commands, the compiler version, all
source hashes, the aggregate result hash, each raw output hash and counts
from `gcov -j -b`. Coverage uses unchanged `linear.cpp`. The script clears
`.gcda` files before each separate measurement. Corpus counts are separate
for each solver; extra counts are separate for each solver and problem.
The contract lists the three shrinking sites, active-row reset and both
300-step exits. The zero-column C 1e3 problem reaches both limit exits.
No claim of convergence or speed is made for that problem.

The Rust tests check all source, input and aggregate output hashes, then
compare every field by bits under `reference/tolerances.json`. They repeat
each fit to check the seed reset. The corpus test writes the three parity
reports. The ignored epsilon-threshold mutation test must fail.
