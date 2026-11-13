# Parity record

What the tests compare, and the result for each solver. The reference is
LIBLINEAR 2.50, built unchanged. Every result below is a count over the saved
reference cases. A match on these cases does not prove a match on other inputs.

The tests parse all 158 cases, check their header counts, load all 316 saved
results, and require equal results for both builds. Each output field must
have a tolerance and a reason. Other tests check input errors and one-row
cases. Solver 2 is compared by bits with all 15 frozen O2 cases. The report
is [reference/parity/solver2.json](../reference/parity/solver2.json). Tests also
check two extra C reference cases, initial weights and prediction ties.
CI runs these commands on pull requests and manual runs.

Heavy checks run by hand. They fuzz the case reader, training and prediction. They prove the weight index bounds, the random stream indices and one feature index rule. They run Miri on the determinism test and on the random and BLAS tests. They mutate the solvers. These checks do not run on every change.

The frozen 2.50 source uses line-search Newton with preconditioned CG.
`src/tron.rs` keeps that algorithm and the C BLAS operation order. The source
map and the older TRON name in the original task notes are explained in the contract.

Solver 2: 15 of 15 cases equal, worst relative error 0, worst 0 ULP against
bit equality. Cases include C of 1e-6 and 1e6, eps of 1e-1 and 1e-8, bias on,
off and free, class weights, a zero feature, a copied row, one row and the
analytic weight `4000/4001`. This is a corpus result, with no speed claim.

Solver 1: 14 of 14 cases equal, worst relative error 0, worst 0 ULP against
bit equality. Solver 3: 13 of 13 cases equal, worst relative error 0, worst
0 ULP against bit equality. The reports are
[solver1.json](../reference/parity/solver1.json) and
[solver3.json](../reference/parity/solver3.json). Cases cover C of 1e-6 and 1e6,
eps of 1e-1 and 1e-8, bias on and off, class weight 10, a zero feature,
a copied row and one row. Solver 1 also covers the analytic weight and its
Newton fallback. Extra C cases check multiclass fits and weight rebuilds.

Solver 0: 14 of 14 cases equal, worst relative error 0, worst 0 ULP against
bit equality. Solver 7: 13 of 13 cases equal, worst relative error 0, worst
0 ULP against bit equality. The reports are
[solver0.json](../reference/parity/solver0.json) and
[solver7.json](../reference/parity/solver7.json). Cases cover C of 1e-6 and
1e6, eps of 1e-1 and 1e-8, class weight 10, bias on and off, zero features,
copied rows and one row. Extra C cases check multiclass fits, initial
weights, free bias and the solver 7 Newton fallback.

The libm table matches 2000 of 2000 exp and 2000 of 2000 log outputs by
bits on glibc 2.39. The test prints the runner's C library version. The C
source, build command and output hash are in `tests/fixtures/`. Another
platform must pass this table test before claiming the same parity.
A changed logarithm fails the model comparison for both solvers. Run the
ignored check alone to see the failure; exit 101 is expected.

```sh
RAYON_NUM_THREADS=2 CARGO_BUILD_JOBS=2 RUST_TEST_THREADS=1 cargo test --locked --test solver07 different_logarithm_matches_reference -- --ignored --exact
```

The safe glibc random stream matches all 10000 C table outputs for each of
seeds 1 and 2. Each train call starts seed 1 and shares the stream across
class fits. A changed generator fails the model comparison. Run its ignored
check alone to see the failure; exit 101 is expected.

```sh
RAYON_NUM_THREADS=2 CARGO_BUILD_JOBS=2 RUST_TEST_THREADS=1 cargo test --locked --test solver13 different_generator_matches_reference -- --ignored --exact
```

BLAS (`daxpy`, `ddot`, `dnrm2`, `dscal`): all nine finite table cases are bit
equal, worst relative error 0, worst 0 ULP. Two extra table cases check
nonfinite values by class for NaN and by bits for other values. Build commands
and the table hash are in [tests/fixtures/README.md](../tests/fixtures/README.md).

The normal test run checks that a reversed dot sum is rejected. To show the
same comparison as a failing command, run this check alone. Exit 101 is expected.

```sh
RAYON_NUM_THREADS=2 CARGO_BUILD_JOBS=2 RUST_TEST_THREADS=1 cargo test --locked --test solver2 reversed_sum_matches_reference -- --ignored --exact
```

Solver 5: 14 of 14 cases equal, worst relative error 0, worst 0 ULP against
bit equality. Solver 6: 14 of 14 cases equal, worst relative error 0, worst
0 ULP against bit equality. Reports are [solver5.json](../reference/parity/solver5.json)
and [solver6.json](../reference/parity/solver6.json). Cases cover C of 1e-6 and
1e6, eps of 1e-1 and 1e-8, class weights, bias on, off and free, zero columns,
copied rows and one row. Six extra C problems match both solvers by bits.
They cover both 20-step line-search exits; see the contract for branch counts.
A changed shrinking comparison fails parity for both solvers. Run the ignored
check alone to show the failure; exit 101 is expected.

```sh
RAYON_NUM_THREADS=2 CARGO_BUILD_JOBS=2 RUST_TEST_THREADS=1 cargo test --locked --test solver56 changed_shrinking_matches_reference -- --ignored --exact
```

Solver 4: 14 of 14 cases equal, worst relative error 0, worst 0 ULP against
bit equality. The report is [solver4.json](../reference/parity/solver4.json).
Cases cover two and three classes, C of 1e-6 and 1e6, eps of 1e-1 and 1e-8,
class weights, bias on and off, zero features, copied rows and one row.
Six extra multiclass C problems match every field by bits. They cover up
to five classes, unbalanced counts, a singleton class, a zero column, copied
rows, C of 1e-3 and 1e3, and the iteration-limit exit. The C qsort table
matches all 10000 seeded double arrays by bits, including ties and signed
zeros. See the contract for branch counts and the fixture build record.
The changed-sort check fails parity. Run the ignored check alone to show
the failure; exit 101 is expected.

```sh
RAYON_NUM_THREADS=2 CARGO_BUILD_JOBS=2 RUST_TEST_THREADS=1 cargo test --locked --test solver4 changed_sort_matches_reference -- --ignored --exact
```

Solver 11: 13 of 13 cases equal, worst relative error 0, worst 0 ULP against
bit equality. Solvers 12 and 13: each has 12 of 12 cases equal, worst
relative error 0, worst 0 ULP against bit equality. Reports are
[solver11.json](../reference/parity/solver11.json),
[solver12.json](../reference/parity/solver12.json) and
[solver13.json](../reference/parity/solver13.json). Cases cover C of 1e-6 and
1e6, eps of 1e-1 and 1e-8, bias on and off, zero features, copied rows,
one row and solver 11 free bias. Six extra C problems match all three
solvers by bits. They cover equal targets, one outlier, p zero and 100,
a copied row, a zero column, and C of 1e-3 and 1e3. The large-C case reaches
both dual iteration limits. The contract gives the branch counts.
A changed dual epsilon threshold fails parity for both solvers 12 and 13.
Run the ignored check alone to show the failure; exit 101 is expected.

```sh
RAYON_NUM_THREADS=2 CARGO_BUILD_JOBS=2 RUST_TEST_THREADS=1 cargo test --locked --test solver11_13 changed_epsilon_threshold_matches_reference -- --ignored --exact
```

Solver 21 matches all 11 frozen cases and six extra C problems by bits,
including rho, with worst 0 ULP against the exact-bit rule. The report is
[solver21.json](../reference/parity/solver21.json). Extra problems cover nu
0.01, 0.5 and 0.99, equal and copied rows, a zero column, one row and
features scaled by 1e3. The scaled case reaches 1000 steps. The contract
records the shrinking, iteration-limit and rho branch counts. A changed
rho update fails parity. Run its ignored test alone; exit 101 is expected.

```sh
RAYON_NUM_THREADS=2 CARGO_BUILD_JOBS=2 RUST_TEST_THREADS=1 cargo test --locked --test solver21 changed_rho_matches_reference -- --ignored --exact
```

## Real datasets

Twenty pairs of a dataset and a solver match the C `train` program, built at
`-O2` with the flags of `reference/BUILD.md`, with worst 0 ULP in the weights, `rho`
and predictions. The datasets are `a9a` (32561 rows, 123 features) and
`rcv1_train.binary` (20242 rows, 47236 features) with solvers 0, 1, 2, 3, 5, 6 and 7;
`satimage.scale` (4435 rows, 36 features, six classes) with solvers 4 and 2;
`cpusmall_scale` (8192 rows, 12 features) with solvers 11, 12 and 13; and `a9a`
with solver 21. All pairs use C 1, eps 0.01 and bias 1, except solver 21: the
C library refuses a bias feature there, so that pair has no bias. The report is
[real_data.json](../reference/parity/real_data.json), with one line for each pair.
It records the shape, both times, the worst ULP and the SHA-256 of the C weight text.
The times are not a speed claim: Rust runs in the unoptimized test profile.

The manifest [real_data_manifest.json](../reference/real_data_manifest.json) holds
the URL and the SHA-256 of each dataset, from the LIBSVM dataset page. The data
stays in a cache folder outside the repository. Set `LIBLINEAR_REAL_DATA_CACHE`
to that folder. If it is unset, the test uses `$HOME/.cache/liblinear-real-data`.
The script takes the folder as its second argument and stops if the folder, after
symbolic links are resolved, is inside the repository. Run the steps in this order:

```sh
export LIBLINEAR_REAL_DATA_CACHE="$HOME/.cache/liblinear-real-data"
python3 -I scripts/real_data_parity.py fetch "$LIBLINEAR_REAL_DATA_CACHE"
RAYON_NUM_THREADS=2 CARGO_BUILD_JOBS=2 python3 -I scripts/real_data_parity.py c "$LIBLINEAR_REAL_DATA_CACHE"
RAYON_NUM_THREADS=2 CARGO_BUILD_JOBS=2 cargo test --locked --test real_data -- --ignored
python3 -I scripts/real_data_parity.py repeat "$LIBLINEAR_REAL_DATA_CACHE"
```

`fetch` stops if a hash differs. `repeat` trains every pair with C again and
requires the same weight text. The test is ignored by default and no CI job runs
it: it reads the datasets that `fetch` downloads. `reference/MANIFEST.sha256`
covers the manifest, so a changed dataset list, URL or hash fails
`scripts/check_reference.py`. The test fails if it compares fewer pairs than the
manifest lists. `python3 -I scripts/test_real_data_scripts.py` checks the cache
folder guard, the converter and the hash pinning without any dataset.

A match on these pairs does not prove a match on other data or on another
C library.
