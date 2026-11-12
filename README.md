# liblinear-rs

A pure Rust port of LIBLINEAR is in progress. The crate has typed inputs and
outputs, a case file reader, and a [contract for all 12 solvers](CONTRACT.md).
Solvers 0 through 7 can train and predict classification models. Solvers
11, 12 and 13 can train and predict support vector regression models.
Solver 21 can train and predict one-class models. All 12 solvers are supported.

The reference uses unchanged LIBLINEAR 2.50 source and its BLAS files.
It covers all 12 solvers with 158 input cases. Each case has saved results
from g++ 13.3.0 at `-O0` and `-O2`. The measured spread is zero. The
[comparison rules](reference/tolerances.json) require exact values for all
output fields. A rule changes only with a new measurement.

## Train and check the ported solvers

Use the stable Rust toolchain. The crate uses edition 2024 and forbids unsafe
code. Typed errors use `thiserror`. The parity report test uses `sha256sum`.
Tests use `serde_json` with float round trips to keep the saved double bits.

```sh
cargo fmt --all -- --check
RAYON_NUM_THREADS=2 CARGO_BUILD_JOBS=2 cargo clippy --locked --all-targets --all-features -- -D warnings
RAYON_NUM_THREADS=2 CARGO_BUILD_JOBS=2 cargo nextest run --locked --all-features
RAYON_NUM_THREADS=2 CARGO_BUILD_JOBS=2 cargo test --locked --doc --all-features
RAYON_NUM_THREADS=2 CARGO_BUILD_JOBS=2 cargo nextest run --locked --no-default-features
cargo deny check
RUSTDOCFLAGS="-D warnings" cargo doc --locked --no-deps --all-features
```

The tests parse all 158 cases, check their header counts, load all 316 saved
results, and require equal results for both builds. Each output field must
have a tolerance and a reason. Other tests check input errors and one-row
cases. Solver 2 is compared by bits with all 15 frozen O2 cases. The report
is [reference/parity/solver2.json](reference/parity/solver2.json). Tests also
check two extra C reference cases, initial weights and prediction ties.
CI runs these commands on pull requests and manual runs.

Heavy checks run by hand. They fuzz the case reader, training and prediction. They prove the weight index bounds, the random stream indices and one feature index rule. They run Miri on the determinism test and on the random and BLAS tests. They mutate the solvers. These checks do not run on every change.

```rust,no_run
use liblinear::{read_problem, train};

let problem = read_problem("reference/corpus/s02_one_row.txt")?;
let model = train(&problem, &problem.parameter)?;
let predictions = model.predict(&problem)?;
assert_eq!(predictions.len(), 1);
# Ok::<(), liblinear::Error>(())
```

`Problem` keeps the solver list and common parameters from the case header.
Sparse row indices start at 1. Rows include the bias feature when bias is
nonnegative. Vector lengths replace the C end markers. `Model` holds the
model fields and the decision values and predictions for the input rows.
See the contract for units, shapes, error rules and the train call.

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
[solver1.json](reference/parity/solver1.json) and
[solver3.json](reference/parity/solver3.json). Cases cover C of 1e-6 and 1e6,
eps of 1e-1 and 1e-8, bias on and off, class weight 10, a zero feature,
a copied row and one row. Solver 1 also covers the analytic weight and its
Newton fallback. Extra C cases check multiclass fits and weight rebuilds.

Solver 0: 14 of 14 cases equal, worst relative error 0, worst 0 ULP against
bit equality. Solver 7: 13 of 13 cases equal, worst relative error 0, worst
0 ULP against bit equality. The reports are
[solver0.json](reference/parity/solver0.json) and
[solver7.json](reference/parity/solver7.json). Cases cover C of 1e-6 and
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
and the table hash are in [tests/fixtures/README.md](tests/fixtures/README.md).

The normal test run checks that a reversed dot sum is rejected. To show the
same comparison as a failing command, run this check alone. Exit 101 is expected.

```sh
RAYON_NUM_THREADS=2 CARGO_BUILD_JOBS=2 RUST_TEST_THREADS=1 cargo test --locked --test solver2 reversed_sum_matches_reference -- --ignored --exact
```

Solver 5: 14 of 14 cases equal, worst relative error 0, worst 0 ULP against
bit equality. Solver 6: 14 of 14 cases equal, worst relative error 0, worst
0 ULP against bit equality. Reports are [solver5.json](reference/parity/solver5.json)
and [solver6.json](reference/parity/solver6.json). Cases cover C of 1e-6 and
1e6, eps of 1e-1 and 1e-8, class weights, bias on, off and free, zero columns,
copied rows and one row. Six extra C problems match both solvers by bits.
They cover both 20-step line-search exits; see the contract for branch counts.
A changed shrinking comparison fails parity for both solvers. Run the ignored
check alone to show the failure; exit 101 is expected.

```sh
RAYON_NUM_THREADS=2 CARGO_BUILD_JOBS=2 RUST_TEST_THREADS=1 cargo test --locked --test solver56 changed_shrinking_matches_reference -- --ignored --exact
```

Solver 4: 14 of 14 cases equal, worst relative error 0, worst 0 ULP against
bit equality. The report is [solver4.json](reference/parity/solver4.json).
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
[solver11.json](reference/parity/solver11.json),
[solver12.json](reference/parity/solver12.json) and
[solver13.json](reference/parity/solver13.json). Cases cover C of 1e-6 and
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
[solver21.json](reference/parity/solver21.json). Extra problems cover nu
0.01, 0.5 and 0.99, equal and copied rows, a zero column, one row and
features scaled by 1e3. The scaled case reaches 1000 steps. The contract
records the shrinking, iteration-limit and rho branch counts. A changed
rho update fails parity. Run its ignored test alone; exit 101 is expected.

```sh
RAYON_NUM_THREADS=2 CARGO_BUILD_JOBS=2 RUST_TEST_THREADS=1 cargo test --locked --test solver21 changed_rho_matches_reference -- --ignored --exact
```

## Check the reference

Use Python 3.12, Bash and g++ 13.3.0. The scripts need no Python packages
or network.

```sh
python3 scripts/check_reference.py
python3 scripts/check_reference.py --self-test
```

The check reads the SHA-256 manifest, checks the source copies, makes the
input cases again and compares both builds. It builds the `-O2` driver and
runs each case twice. Both runs must match the saved `-O2` file byte for byte.
The analytic case checks solvers 1 and 2 against `4000/4001` with an absolute
limit of `1e-4`.

`bash scripts/freeze_reference.sh` replaces the saved reference files.
See [the build record](reference/BUILD.md),
[the source record](vendor/liblinear/PROVENANCE.md),
[the case format](reference/README.md), and
[the measured spread](reference/spread.json).

## Licence

BSD-3-Clause, the licence of the original LIBLINEAR. See [LICENSE](LICENSE)
and [NOTICE](NOTICE). The upstream terms are copied unchanged in
[THIRD-PARTY-NOTICES](THIRD-PARTY-NOTICES).
