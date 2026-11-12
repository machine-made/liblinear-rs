# LIBLINEAR port contract

This contract uses LIBLINEAR 2.50 and the saved reference results. The Rust
crate has types, a case reader, and training and prediction for solvers
0 through 7, 11 through 13, and 21. All 12 solver numbers are supported.

## Common call and inputs

Training uses this Rust signature:

```rust,ignore
pub fn train(problem: &Problem, parameter: &Parameter) -> Result<Model, Error>;
```

The call borrows both inputs and returns an owned model. It does not change
the input rows. Set `parameter.solver_type` for one call. The `solvers` list
in a case gives the calls to make, in order. `problem.parameter` holds the
shared header values and selects the first solver. A caller can copy it and
select each other solver. The explicit train parameter is the source of
training options.

Let `m` be `problem.rows.len()`, `f` be `problem.nr_feature`, and `n` be
`f + 1` when bias is nonnegative, or `f` otherwise. Let `k` be the class
count. Let `d` be 1 when `k == 2` and the solver is not 4, or `k` otherwise.

| Rust input | Shape, units and meaning |
| --- | --- |
| `Problem.rows` | `m` sparse rows of `Feature { index: usize, value: f64 }`; indices start at 1 and increase strictly; absent values are zero |
| `Problem.labels` | `m` finite `f64` values; class labels for 0 through 7, targets for 11 through 13, ignored for 21 |
| `Problem.nr_feature` | `f` input features before bias; includes features that are zero in all rows |
| `Problem.bias` | Finite `f64`; negative omits bias; otherwise append feature `(f + 1, bias)` to each row |
| `Problem.solvers` | Nonempty list of `SolverType`, in case order |
| `Problem.parameter` | Header values in a `Parameter`; its solver is the first listed solver |
| `Parameter.solver_type` | One of 0, 1, 2, 3, 4, 5, 6, 7, 11, 12, 13, 21 |
| `Parameter.c` | C; positive, finite cost multiplier; scalar |
| `Parameter.eps` | Positive, finite stop limit; scalar; uses the selected upstream stop test |
| `Parameter.class_weights` | Ordered `(i32, f64)` pairs; label and positive finite multiplier for C; count is the vector length; used by 0 through 7 |
| `Parameter.p` | Finite loss width, in target units; used by 11 through 13; must be nonnegative for 11 |
| `Parameter.nu` | Finite one-class fraction, greater than zero and at most 1; used by 21 |
| `Parameter.regularize_bias` | `bool`; true applies the bias penalty; false requires bias 1 and solver 0, 2, 5, 6 or 11 |
| `Parameter.init_sol` | `None` in the case reader, as in the driver; an initial vector has `n * d` weights and is allowed only for 0, 2 and 11 |
| `Parameter.w_recalc` | False in the case reader, as in the driver; true is allowed only for 1 and 3 |

Counts and indices are unitless. The file reader accepts counts from 1 to
2147483646. Features and targets have the units supplied by the caller; the
library does not rescale them. A weight times its feature has score units.
C, class weights, nu and bias are unitless inputs. Regression scores have
target units. Classification scores, rho and eps use the upstream score and
stop-test scales; eps is not a limit on prediction error. Classification
labels must be integers in the i32 range for train. The file reader keeps
finite labels as f64 so that it can also read regression cases.

The case format is in [reference/README.md](reference/README.md). Every header
field is required. Blank lines are skipped. Unknown keys, repeated keys,
extra scalar values, invalid counts, nonfinite numbers, unordered indices,
indices outside 1 through f, and a wrong row count return `Error`.
Vectors replace the C end markers. Bias is appended once, including when
its value is zero. The reader checks the file shape; it does not check the
solver-specific parameter rules. Thus a file with finite C of zero can be
read, but train must reject it. A missing file returns `Error::Io` and asks
the caller to check the file and read access. Input errors give the line
number when known and state the correction.

## Common outputs

The model owns these values. It also holds decision values and predictions
for all input rows, in input order, as saved by the reference driver.

| Golden field | Rust field | Shape, units and meaning |
| --- | --- | --- |
| `runs` | One `Model` per call | Same order as the case solvers |
| `solver` | `parameter.solver_type` | Exact solver number |
| `C` | `parameter.c` | Exact copy of the cost scalar |
| `eps` | `parameter.eps` | Exact copy of the stop scalar |
| `bias` | `bias` | Exact copy of the input bias |
| `nr_class` | `nr_class: usize` | k; distinct class count for 0 through 7, 2 for 11 through 13 and 21 |
| `nr_feature` | `nr_feature: usize` | f; excludes the bias feature |
| `labels` | `labels: Option<Vec<i32>>` | k labels for classification; None for 11 through 13 and 21 |
| `weights` | `weights: Vec<f64>` | `n * d`; feature first, then class; last feature is bias when present |
| `rho` | `rho: Option<f64>` | One score threshold for 21; None for every other solver |
| `decision_values` | `decision_values: Vec<Vec<f64>>` | `m * d` scores; for 21 the score already has rho subtracted |
| `predictions` | `predictions: Vec<f64>` | m class labels, target values, or one-class signs |

Class labels follow their first occurrence in the rows. For two classes
with labels -1 and +1, put +1 first. For binary solvers other than 4, a
positive score selects the first label; zero or a negative score selects
the second. Solver 4 with two classes keeps two score columns but uses the
same first-score sign rule as upstream. With more than two classes, select
the greatest score; a tie selects the first label with that score. With one
class, return that label. Regression returns the sole decision value.
Solver 21 returns +1 for a positive score and -1 otherwise.

## Parameter errors and empty inputs

These are the exact strings from `check_parameter` in
`vendor/liblinear/linear.cpp`, in check order. They are source text for
compatibility; the Rust error must also state the correction in plain words.

| Condition | Upstream text | Rust correction |
| --- | --- | --- |
| eps <= 0, all solvers | `eps <= 0` | Set eps to a number greater than zero. |
| C <= 0, all solvers | `C <= 0` | Set C to a number greater than zero. |
| p < 0, solver 11 | `p < 0` | Set p to zero or a positive number. |
| bias >= 0, solver 21 | `prob->bias >=0, but this is ignored in ONECLASS_SVM` | Set bias to a negative number for solver 21. |
| Free bias but bias != 1 | `To not regularize bias, must specify -B 1 along with -R` | Set bias to 1 to omit its penalty. |
| Free bias with an unsupported solver | `-R option supported only for solver L2R_LR, L2R_L2LOSS_SVC, L1R_L2LOSS_SVC, L1R_LR, and L2R_L2LOSS_SVR` | Apply the bias penalty or select solver 0, 2, 5, 6 or 11. |
| Unknown solver | `unknown solver type` | The Rust enum returns `solver N does not exist.` |
| Initial weights with an unsupported solver | `Initial-solution specification supported only for solvers L2R_LR, L2R_L2LOSS_SVC, and L2R_L2LOSS_SVR` | Remove the initial weights or select solver 0, 2 or 11. |
| Weight recalculation with an unsupported solver | `Recalculating w in the end is only for dual solvers for L2-regularized L1/L2-loss SVM` | Turn off weight recalculation or select solver 1 or 3. |

`SolverType::try_from` rejects unknown numbers before a train call. The case
reader sets initial weights to None and weight recalculation to false, as
the driver does. The Rust train contract also rejects nonfinite inputs,
nonpositive class weights, invalid nu for 21, wrong vector sizes, and empty
problems. These added shape and range checks prevent invalid memory access
in the upstream API; they do not change any saved case.

For all 12 solvers an empty problem returns an error: "Add at least one data
row." The case reader rejects rows of zero and missing data rows. The
reference driver also rejects an empty case. Upstream train does not define
a safe result for an empty problem, so no numeric result is promised.
An empty feature list in a row is allowed when the declared feature count
is positive. A one-row problem is valid and returns one row of decision
values and one prediction. Its weights must match the saved one-row case;
no new numeric shortcut is specified here.

## Bounds errors

`Error::Index` means an index is outside its array. The message gives the
index and the length. Use an index below that length.

`Error::Length` means two slices do not have the same length. The message
gives both lengths. Use two slices of equal length.

`Error::Overflow` means a size calculation overflowed. Use fewer rows,
classes, or features.

`Error::Count` means a row, class, or feature count does not fit in 32 bits.
Use fewer rows, classes, or features.

The dual, L2, L1, multiclass and one-class solvers return these errors when
a length differs, an index is outside its array, or a size calculation
overflows.

A feature index of zero returns an input error. Start feature indices at 1.

A line number that overflows returns an input error. Use a shorter file.

`GlibcRand::new` returns an input error when a seed step falls outside the
31-bit range. Report that failure.

## Solver 0: L2R_LR

`pub fn train(problem: &Problem, parameter: &Parameter) -> Result<Model, Error>`
with `SolverType::L2rLr`. L2 penalty and logistic loss. Uses C, eps, class
weights and the bias option. Initial weights and free bias are allowed.
Uses the common m-row inputs and n*d weight output. One row gives k=1,
d=1, that row's label, n weights, null rho and one prediction of the label.
Empty input returns the common error. All common parameter errors apply;
the p, nu and solver-21 bias checks do not apply.

## Solver 1: L2R_L2LOSS_SVC_DUAL

`pub fn train(problem: &Problem, parameter: &Parameter) -> Result<Model, Error>`
with `SolverType::L2rL2lossSvcDual`. L2 penalty and squared hinge loss, dual
form. Uses C, eps, class weights and penalized bias. Weight recalculation
is allowed; initial weights and free bias are rejected. Uses the common
m-row inputs and n*d weights. One row gives k=1, d=1, that row's label,
n weights, null rho and one prediction of the label. Empty input returns
the common error. The p, nu and solver-21 bias checks do not apply.

## Solver 2: L2R_L2LOSS_SVC

`pub fn train(problem: &Problem, parameter: &Parameter) -> Result<Model, Error>`
with `SolverType::L2rL2lossSvc`. L2 penalty and squared hinge loss, primal
form. Uses C, eps, class weights and the bias option. Initial weights and
free bias are allowed. Weight recalculation is rejected. Uses the common
m-row inputs and n*d weights. One row gives k=1, d=1, that row's label,
n weights, null rho and one prediction of the label. Empty input returns
the common error. The p, nu and solver-21 bias checks do not apply.

## Solver 3: L2R_L1LOSS_SVC_DUAL

`pub fn train(problem: &Problem, parameter: &Parameter) -> Result<Model, Error>`
with `SolverType::L2rL1lossSvcDual`. L2 penalty and hinge loss, dual form.
Uses C, eps, class weights and penalized bias. Weight recalculation is
allowed; initial weights and free bias are rejected. Uses the common
m-row inputs and n*d weights. One row gives k=1, d=1, that row's label,
n weights, null rho and one prediction of the label. Empty input returns
the common error. The p, nu and solver-21 bias checks do not apply.

## Solver 4: MCSVM_CS

`pub fn train(problem: &Problem, parameter: &Parameter) -> Result<Model, Error>`
with `SolverType::McsvmCs`. Joint multiclass SVM with L2 penalty. Uses C,
eps, class weights and penalized bias. Initial weights, free bias and
weight recalculation are rejected. Uses m rows, n*k weights and m*k scores,
also when k=2. One row gives k=1, d=1, that row's label, n weights, null
rho and one prediction of the label. Empty input returns the common error.
The p, nu and solver-21 bias checks do not apply.

## Solver 5: L1R_L2LOSS_SVC

`pub fn train(problem: &Problem, parameter: &Parameter) -> Result<Model, Error>`
with `SolverType::L1rL2lossSvc`. L1 penalty and squared hinge loss. Uses C,
eps, class weights and the bias option. Free bias is allowed; initial
weights and weight recalculation are rejected. Uses the common m-row inputs
and n*d weights. One row gives k=1, d=1, that row's label, n weights, null
rho and one prediction of the label. Empty input returns the common error.
The p, nu and solver-21 bias checks do not apply.

## Solver 6: L1R_LR

`pub fn train(problem: &Problem, parameter: &Parameter) -> Result<Model, Error>`
with `SolverType::L1rLr`. L1 penalty and logistic loss. Uses C, eps, class
weights and the bias option. Free bias is allowed; initial weights and
weight recalculation are rejected. Uses the common m-row inputs and n*d
weights. One row gives k=1, d=1, that row's label, n weights, null rho and
one prediction of the label. Empty input returns the common error.
The p, nu and solver-21 bias checks do not apply.

## Solver 7: L2R_LR_DUAL

`pub fn train(problem: &Problem, parameter: &Parameter) -> Result<Model, Error>`
with `SolverType::L2rLrDual`. L2 penalty and logistic loss, dual form. Uses
C, eps, class weights and penalized bias. Initial weights, free bias and
weight recalculation are rejected. Uses the common m-row inputs and n*d
weights. One row gives k=1, d=1, that row's label, n weights, null rho and
one prediction of the label. Empty input returns the common error.
The p, nu and solver-21 bias checks do not apply.

## Solver 11: L2R_L2LOSS_SVR

`pub fn train(problem: &Problem, parameter: &Parameter) -> Result<Model, Error>`
with `SolverType::L2rL2lossSvr`. L2 penalty and squared loss outside width p,
primal form. Uses C, eps, p and the bias option; ignores class weights.
Initial weights and free bias are allowed. Weight recalculation is rejected.
Inputs are m targets and rows; k=2, d=1, labels and rho are null. Outputs
are n weights, m scores and m target predictions. One row keeps these sizes
with m=1. Empty input returns the common error. Negative p is rejected;
the nu and solver-21 bias checks do not apply.

## Solver 12: L2R_L2LOSS_SVR_DUAL

`pub fn train(problem: &Problem, parameter: &Parameter) -> Result<Model, Error>`
with `SolverType::L2rL2lossSvrDual`. L2 penalty and squared loss outside width
p, dual form. Uses C, eps, p and penalized bias; ignores class weights.
Initial weights, free bias and weight recalculation are rejected. Inputs
are m targets and rows; k=2, d=1, labels and rho are null. Outputs are n
weights, m scores and m target predictions. One row keeps these sizes with
m=1. Empty input returns the common error. Upstream does not reject negative
p for this solver; the nu and solver-21 bias checks do not apply.

## Solver 13: L2R_L1LOSS_SVR_DUAL

`pub fn train(problem: &Problem, parameter: &Parameter) -> Result<Model, Error>`
with `SolverType::L2rL1lossSvrDual`. L2 penalty and linear loss outside width
p, dual form. Uses C, eps, p and penalized bias; ignores class weights.
Initial weights, free bias and weight recalculation are rejected. Inputs
are m targets and rows; k=2, d=1, labels and rho are null. Outputs are n
weights, m scores and m target predictions. One row keeps these sizes with
m=1. Empty input returns the common error. Upstream does not reject negative
p for this solver; the nu and solver-21 bias checks do not apply.

## Solver 21: ONECLASS_SVM

`pub fn train(problem: &Problem, parameter: &Parameter) -> Result<Model, Error>`
with `SolverType::OneclassSvm`. One-class SVM. Uses eps and nu; C is checked
but is not used in the fit. Ignores row labels, p and class weights. Bias
must be negative. Initial weights, free bias and weight recalculation are
rejected. Inputs are m rows; k=2, d=1, n=f, and labels are null. Outputs are
n weights, numeric rho, m scores after rho subtraction, and m signs. One
row keeps these sizes with m=1. Empty input returns the common error. The
one-class bias error applies; the negative-p error does not apply.

## Comparison rule and limits

[reference/tolerances.json](reference/tolerances.json) names every field,
including containers, null values and numeric leaves. It lists the fields
for each solver. All rules are exact. Counts, labels and predictions must
be equal. Finite floating-point values must have equal f64 bits, including
the sign of zero. Array sizes and order must be equal. No field may be
skipped because it has no tolerance entry. Null rho is required outside
solver 21; null labels are required for regression and one-class results.

The reason is the zero absolute and relative spread in
[reference/spread.json](reference/spread.json) between the -O0 and -O2 builds.
Use the same operations in the same order as the original. A tolerance
changes only with a new measurement. These two builds used the same
compiler and platform; their equality does not prove a Rust solver matches.
The later solver checks must measure that match. Solvers 0, 6 and 7 use exp
and log. Their checks must measure those results on the target platform.
The reference uses the C random number generator with seed 1 before each
train call. Later ports must check random order as well as numeric results.

All 12 solvers are implemented.
`Model::decision_values(&Problem)` and `Model::predict(&Problem)` return owned
vectors for models from all 12 solvers. A one-class model without rho returns
an error. A classification model without labels returns an error from predict.
They use each
supplied row as it is. Include the bias feature in those rows, as the reader does. Features above the model size
are skipped, as in upstream `predict_values`. Training also saves both vectors
in the model. Initial weights use the common feature-then-class layout.

## Solver 2 source map

The frozen 2.50 source uses `NEWTON::newton` and `NEWTON::pcg` in
`newton.cpp`. It uses a line search, not the older TRON trust-region method.
The original task notes names TRON, `trcg` and eta/sigma constants, but those names and
constants are absent from this source. `src/tron.rs` ports the frozen NEWTON
code. It keeps the PCG constant 0.01, CG limit 0.5, line search constant 0.01,
20 line search steps and 1000 Newton steps. These source values take priority
so that the port can match the frozen results.

| C routine or method | Rust call | Scope |
| --- | --- | --- |
| `daxpy_` | `blas::daxpy(a, x, y)` | Equal slice lengths; groups of four, then the tail |
| `ddot_` | `blas::ddot(x, y)` | Equal slice lengths; sum each group of five, then add it to the total |
| `dnrm2_` | `blas::dnrm2(x)` | Scaled norm; visit entries from last to first |
| `dscal_` | `blas::dscal(a, x)` | Groups of five, then the tail |
| `NEWTON::newton`, `pcg` | `tron::newton`, `pcg` | Same stop tests and sum order |
| `l2r_erm_fun::fun`, `linesearch_and_update` | `Objective::fun`, `linesearch_and_update` | Same loss and penalty sums |
| `l2r_erm_fun::Xv`, `XTv` | `xv`, `Objective::xtv` | Sparse dot and transpose loops; XTv is not called by solver 2 |
| `l2r_l2_svc_fun::grad`, `Hv` | `Objective::grad`, `hv` | Same active rows and loop order |
| `get_diag_preconditioner`, `subXTv` | `Objective::get_diag_preconditioner`, `sub_xtv` | Same sparse updates |
| `group_classes`, `train`, `predict_values` | `group_classes`, `train`, `Model` methods | Stable class groups and feature-then-class weights |

The four safe BLAS calls cover unit strides, which are all the strides used
by this solver. Empty slices return zero for dot and norm; axpy and scale do
nothing. The calls do not use FMA, a library BLAS or reordered sums. The C table
checks lengths 0, 1, 4, 5, 6 and 9, large and small scales, subnormal values,
and signed zeros by bits. For the extra nonfinite BLAS checks, a NaN must stay
a NaN; IEEE arithmetic does not fix its sign or payload. This does not change
the finite solver comparison rule. Training rejects nonfinite inputs.

The solver test checks all 15 frozen solver 2 cases with the exact rules and
writes [reference/parity/solver2.json](reference/parity/solver2.json). The report
records input hashes, a candidate hash, counts and maximum errors. Candidate
hashes use the compact JSON encoding from `serde_json`, with sorted object
keys and no final newline. Output counts cover numeric leaves. Containers,
nulls and all field names are also checked. The tests use a comparator for the
saved nested model format, which includes nulls and objects inside arrays.
The generic port parity script does not accept that format.

Two extra C reference cases check multiclass free bias, repeated class weights,
a missing weight label and the binary label swap. A reversed-dot mutation uses
the same training and comparison calls and must fail. The separate ignored
check runs that mutation as an expected failing command. No speed claim is made.

The tests read all 158 case files, compare row and feature counts with the
headers, load all 316 golden files, and compare both saved builds by value,
floating-point bits and file bytes. They check model shapes and require a
tolerance and a reason for every output field. These checks prove the
reader and saved-data contract. The solver 2 check also measures parity on its
15 cases. These results do not prove correctness outside the corpus, measure
speed, or claim solver results for nonfinite inputs.

## Solver 1 and 3 source map

`src/dual_l1l2_svc.rs` ports `solve_l2r_l1l2_svc` from the frozen
`linear.cpp`. It keeps the shuffle, active set, bound checks, gradient stop
test, 1e-12 update limit and 300-step limit. The original task notes names `lower_bound`,
but the source has no such array: each alpha has a lower bound of zero.
The upper bound is infinity for solver 1 and class C for solver 3. The
source uses `0.5/C`, not a rewritten `1/(2*C)`, for solver 1's diagonal.
Sparse dot, squared norm and axpy loops visit features in input order.
The objective sum and support count only feed the source print callback;
the Rust solver omits these print-only values.

The original task notes does not name the fallback in `train_one`. At 300 steps, solver 1
keeps the dual weights and runs the solver 2 Newton method. It uses the
class-scaled primal stop limit times 0.1. Solver 3 keeps its dual result at
that limit. The frozen `analytic_l2svc` case covers the solver 1 fallback;
the C fixture driver counts one switch for that case. The analytic weight
is bit equal to `0.99975006248437892`.

`rand::GlibcRand::new(seed)` and `next_u31()` copy the glibc `TYPE_3`
stream: 31 words, separation 3, signed seed setup, wrapping addition and
310 discarded outputs. Seed zero selects seed one. `new` returns an error
when a seed step falls outside the 31-bit range. `train` starts seed 1
once per call, as in the frozen driver. Multiclass fits share this stream;
they do not reset it for each class. There is no global random state.
`train_with_rand` takes a supplied stream for the mutation check.

The C table in `tests/fixtures/rand_table.json` holds 10000 outputs for each
of seeds 1 and 2 from glibc 2.39. Its C source, build command and SHA-256 are
stored next to it. The first seed-1 output is 1804289383. The tests compare
every table entry and check that zero and one seeds give the same stream.

The model tests use the same nested comparator as solver 2 and require the
unchanged rules in `reference/tolerances.json`. All 14 solver 1 cases and
all 13 solver 3 cases match by bits, with worst relative error 0 and worst
0 ULP. The reports are `reference/parity/solver1.json` and `solver3.json`.
Each report stores per-case hashes, numeric counts and maximum errors.
The cases cover zero features, copied rows, one row, C of 1e-6 and 1e6,
eps of 1e-1 and 1e-8, bias on and off, and class weight 10.

The frozen driver always leaves `w_recalc` false, so no frozen corpus case
covers that branch. Extra C cases in `tests/fixtures/dual_*.json` cover it
for both solvers. They use the frozen source with the flag on and off,
and check multiclass stream use, repeated class weights, an absent weight
label, binary label order and zero bias. At least one rebuild must change
weight bits, so an omitted rebuild cannot pass. A changed random generator
must fail the same model comparison. Its ignored test shows that failure
as a separate command. No speed claim is made.

## Solvers 0 and 7: source map and measured parity

Solver 0 uses `l2r_lr.rs` for `l2r_lr_fun` and its `l2r_erm_fun` helpers:
objective, gradient, Hessian-vector product, diagonal preconditioner and
line search. The shared `tron::Objective` interface sends both primal
losses through the same Newton and PCG code. All sums keep the frozen order.
The sparse dot and axpy helpers are the same as for solver 2. `exp` and
`log` map to `f64::exp` and `f64::ln`. No `log1p` or fused operation is used.

Solver 7 uses `dual_lr.rs` for `solve_l2r_lr_dual`. It keeps the pair
`alpha` and `C - alpha`, the seed-1 glibc shuffle, the 300 outer steps,
`innereps` of 1e-2, its lower limit of min(1e-8, eps), and the inner loop
condition `inner_iter <= 100`. The frozen source calls its inner step
factor `eta` and sets it to 0.1. The original task notes also names `xi`, `lambda` and a
trust-region method. Those are not variables or methods in this source.
The source uses line-search Newton. If solver 7 reaches 300 dual steps,
`train_one` keeps its weights and starts primal logistic Newton with the
scaled primal tolerance times 0.1. The final dual objective only feeds the
print callback, so the port omits that calculation.

`tests/fixtures/libm_table.c` prints 2000 inputs and outputs for each of
`exp` and `log` with `%.17g`. A fixed seed of 1 selects samples. The exp
inputs include 1e-300 through 700, negative inputs through -700, values
near zero, and signed zero. The log inputs span 1e-300 through 1e300 and
include values near 1. The JSON table stores the build command, compiler,
C library and SHA-256 of the raw C output. The C build uses g++ 13.3.0
with `-O2` and glibc 2.39. Rust matches all 2000 exp and 2000 log outputs
by bits on glibc 2.39. The test prints the runner's C library version.
This is a table result; another platform must pass the same check.

All 14 solver 0 cases and all 13 solver 7 cases match every frozen field
by bits, with worst relative error 0 and worst 0 ULP. The reports are
`reference/parity/solver0.json` and `solver7.json`. Cases cover C of 1e-6
and 1e6, eps of 1e-1 and 1e-8, class weight 10, bias 1 and -1, copied rows,
zero features and one row. Solver 0 also covers free bias.

Extra C cases in `tests/fixtures/lr_*.json` check multiclass fits, shared
random stream use, repeated weights, an absent weight label, binary label
order, zero bias, free bias and initial weights. The fallback case uses
C of 1e6 and eps of 1e-16; its C driver counts two primal fallbacks.
`train_with_log` supplies a logarithm for the mutation check. Replacing
`ln(x)` with `log2(x) * LN_2` fails the same model comparison for both
solvers. The ignored test shows this failure when run alone. Tolerance
rules and frozen files are unchanged. No speed claim is made.

## Solvers 5 and 6: source map and measured parity

`src/l1r_l2_svc.rs` ports `solve_l1r_l2_svc`. `src/l1r_lr.rs` ports
`solve_l1r_lr`. Each fit makes private feature columns in grouped row order.
Solver 5 multiplies these columns by y. It keeps b, the column squared sums,
the gradient norm, the active set and its reset. Both solvers use the glibc
random stream from the common train call. A multiclass fit shares that stream.
The stop limit is `eps * max(min(pos, neg), 1) / l`, as in `train_one`.
A free bias is the last column; its gradient has no L1 term and it is never
removed from the active set.

The original task notes names M and beta for solver 5. The frozen source has neither name.
It uses `Gmax_old / l` as the shrinking bound and multiplies each rejected
step and delta by 0.5. Solver 5 has 1000 outer steps, sigma 0.01, a Hessian
floor of 1e-12, an update limit of 1e-12 and 20 line-search steps.
The original task notes names `xjpos_sum` and `x_min` for solver 6. Neither exists in the
frozen source. Solver 6 uses `xjneg_sum`, the exponential cache, tau and D.
It has 100 Newton steps, 1000 inner steps, nu 1e-12, sigma 0.01, an initial
inner limit of 1, a step bound of [-10, 10] and 20 line-search steps.
An inner solve with one step multiplies the inner limit by 0.25.
`exp` and `log` use `f64::exp` and `f64::ln`. The port keeps the sum order
and the cache reset after a failed line search. Print-only objective sums,
support counts and restoration of the private columns are omitted.

All 14 solver 5 cases and all 14 solver 6 cases match every saved field
by bits, with worst relative error 0 and worst 0 ULP. Reports are
`reference/parity/solver5.json` and `reference/parity/solver6.json`.
The cases include C of 1e-6 and 1e6, eps of 1e-1 and 1e-8, class weight 10,
bias 1 and -1, free bias, zero columns, a copied row and one row.
`train_with_shrinking` selects a changed comparison for the mutation check.
It changes `Gp > bound` to `Gp < bound`; the other comparison stays the same.
This mutation fails the field comparison for both solvers. The ignored test
shows the solver 5 failure as a separate command.

### Extra C problems and branch counts

`tests/fixtures/l1_reference.cpp` calls the unchanged vendored source through
the frozen reader and writer. It prints doubles with `%.17g` and starts
seed 1 for each call. The six `l1_*.txt` inputs cover two features, zero
columns, one class, a copied row, C of 1e-3 and C of 1e3. The zero-column
case also covers three classes, zero bias, repeated weights and an absent
weight label. The one-class and copied-row cases use free bias. All six
problems match both Rust solvers by bits for every field.

`l1_extra.json` stores the inputs' hashes and the C results.
`l1_extra.build.json` stores both build commands, compiler and C library,
the SHA-256 of the result file and of each raw C output, and coverage counts.
The extra outputs from `-O0 --coverage` and `-O2` are equal by bytes.
The fixture README gives the build, run and coverage commands. No reference
corpus, golden file or tolerance was changed.

GCC coverage on unchanged `vendor/liblinear/linear.cpp` measures these
branches. Counts are branch entries. For the solver 6 cache reset, divide the
execution count at line 2110 by the row count:

| Branch | Frozen corpus | Six extra problems |
| --- | ---: | ---: |
| Solver 5 removes a zero weight, line 1599 | 184 | 14 |
| Solver 6 removes a zero weight in the outer loop, line 1917 | 768 | 406 |
| Solver 6 removes a zero trial weight in the inner loop, line 1991 | 3 | 128 |
| Solver 5 resets b after 20 rejected steps, line 1700 | 0 | 3 |
| Solver 6 resets its cache after 20 rejected steps, line 2110 / l | 0 | 6 |

`l1_C_large.txt` reaches the solver 5 line-search exit three times.
It has 12 rows, six features with different scales, C of 1000 and eps of
1e-16. Its data uses Python's `random.Random(106)`: draw six scales as
`10 ** uniform(-5, 5)`, then draw each row's six values as
`uniform(-1, 1) * scale`; labels alternate -1 and +1, starting with -1.
It uses bias 1 and keeps its penalty. It reaches the 1000-step limit.
`l1_one_class.txt` reaches the solver 6 line-search exit six times.
It has three rows, label 9, free bias and eps of 1e-30. It reaches the
100-step Newton limit. These are finite input and output checks of the
source behavior at its step limits, not claims of convergence.

For all three shrinking sites, the corpus reaches zero weights with
`Gp < 0`, zero weights with `Gn > 0`, weights kept at zero, weights removed,
and positive and negative nonzero weights. It reaches both outcomes of the
first bound comparison. The corpus also reaches both outcomes of the second
bound comparison for solver 5. It does not reach a false second comparison
for solver 6 after the first is true. `l1_C_large.txt` adds that branch in
both the outer and inner loops. The corpus and extra free-bias cases reach
the path which bypasses shrinking. The corpus reaches active-set resets
in solver 5 and in solver 6's inner loop; the large-C extra case reaches
43 inner resets. No speed claim is made.


## Solver 4 source map and proof

`src/mcsvm_cs.rs` ports `Solver_MCSVM_CS::Solve`, `solve_sub_problem`
and `be_shrunk`. It keeps the alpha and gradient arrays, class indices,
per-row active sizes, swaps, and sum order. The step limit is 100000.
The shrinking stop limit starts at `max(10 * eps, 1)` and halves on a
reset, with eps as its lower bound. Gradient gaps and weight changes use
1e-12, as in the source. C is set for each class after all class weights.
Only print-only objective sums and support counts are omitted.

`train` groups rows by class in first-label order and keeps row order
within each class. The -1, +1 label pair is put in +1, -1 order. The joint
solver receives class indices from zero, the full eps value, and one glibc
random stream with seed 1. A two-class solver 4 problem takes this same
joint path in upstream `train`; it does not call `train_one`. Its weights
have two columns. Weights use feature order, then class order:
`weights[feature * nr_class + class]`. The binary prediction still uses
the sign of the first score, as in upstream `predict_values`.

The original task notes says one block per class as well as feature-major order. The
source resolves this: adjacent weights belong to the same feature and
different classes. It also passes eps directly; the binary solvers' eps
scale does not apply here. No reference input, result or tolerance changed.

The source sorts doubles in decreasing order with `compare_double`.
Rust uses a stable `sort_by` with the same two comparisons. Equal doubles
are interchangeable in the subproblem sums. NaN does not occur in the
checked subproblems. Positive and negative zero compare equal but have
different bits; the stable Rust sort keeps their input order, as glibc
qsort does in the saved table. The test requires the bits to match.
`tests/fixtures/mcsvm_sort.c` supplies 10000 seeded arrays from a palette
of 16 doubles. The palette includes ties, both zero signs, positive and
negative subnormal values, normal values and the largest finite values.
`mcsvm_sort.json` stores each sorted array as palette indices, with no
loss of bits. `mcsvm_sort.build.json` stores the commands and hashes.

All 14 corpus cases match every field by bits, worst relative error 0,
worst 0 ULP. The report is `reference/parity/solver4.json`. Cases include
two and three classes, C of 1e-6 and 1e6, eps of 1e-1 and 1e-8, class
weights, bias on and off, one row, a copied row and zero features.
`train_with_mcsvm_sort` accepts a sort function for the parity check. The
changed comparator exists only in `tests/solver4.rs`. It changes `a > b`
to `a < b` and keeps the second test.
The ignored test fails on `s04_C_large`, at `decision_values[0][0]`:
actual bits 0x3ff2341896bce3d8, reference bits 0x3ff2aa7397477cfb.
The normal test requires this mismatch. No speed claim is made.

### Extra multiclass C problems and branch counts

`tests/fixtures/mcsvm_reference.cpp` uses the frozen reader and writer and
the unchanged vendored solver. It prints doubles with `%.17g` and starts
seed 1 for each call. `mcsvm_reference.py` builds and runs it without the
network and reads GCC coverage counts from that program. It saves six
problems: three classes with one singleton class; four classes; five
classes with counts 5, 3, 2, 1, 1; a copied row, including a copy with a
different label; a zero column with C 1e-3; and five unbalanced classes
with C 1e3 and eps 1e-30. All use bias 1 with its penalty.

The six O0 coverage outputs and O2 outputs match by bytes. Each Rust model
matches all C fields by bits. `mcsvm_extra.json` holds the results.
`mcsvm_extra.build.json` holds the build commands, compiler version,
output SHA-256, raw output hashes, input hashes via the results, and
coverage counts for the corpus and each extra problem. `{repo}` in a
command means the absolute repository path. The fixture README gives
the command that builds and runs all six cases again.

These are execution counts from the unchanged C source. Class removal
can remove more than one class in a row visit, so its two sites are shown
apart. A reset count is one entry, independent of the number of rows.

| Source path | Source line | Corpus | Six extra problems |
| --- | ---: | ---: | ---: |
| Remove the first class in a shrinking step | 753 | 9337 | 871 |
| Remove another class in that step | 767 | 557 | 893 |
| Remove a row with at most one active class | 774 | 9194 | 179 |
| Reset all active rows and classes | 825 | 85 | 98 |
| Exit at the 100000-step limit | 839 | 0 | 1 |

The extra C 1e3 problem reaches the iteration limit once. This is a check
of the source behavior at its limit, not a claim of convergence.


## Solvers 11, 12 and 13 source map and proof

`src/l2r_l2_svr.rs` ports `l2r_l2_svr_fun` and the methods it inherits from
`l2r_l2_svc_fun` and `l2r_erm_fun`. The loss uses `d = wx - y`. It is zero
when `-p <= d <= p`, `C * (d + p) * (d + p)` below that range, and
`C * (d - p) * (d - p)` above it. The gradient selects active rows in input
order. The Hessian product and its diagonal use that same row order.
The shared Newton code in `src/tron.rs` uses the existing Rust BLAS calls.
No new BLAS calls or dependencies were added.

`src/dual_svr.rs` ports `solve_l2r_l1l2_svr`. Solver 12 uses lambda `0.5 / C`
and an infinite upper bound. Solver 13 uses lambda zero and upper bound C.
The port keeps beta, row norms, the glibc shuffle with seed 1, all three
shrinking sites, active-set resets, direction and beta updates, and the
relative gradient stop test. The direction threshold is 1e-12. The step
limit is 300. Only the print-only objective sums and support counts are
omitted. C min and max comparisons retain their NaN and signed-zero rules.

The regression path of `train` keeps the input rows and targets in their
original order. It does not group labels or apply class weights. The
Newton stop limit is eps without the class-count scale. Solver 12 switches
to Newton at 300 steps, with the current dual weights and `eps * 0.001`.
Solver 13 returns the dual weights at that limit. Solver 11 accepts initial
weights and free bias. The model has one weight vector, class count 2,
null labels and rho, one score per row, and that score as its prediction.

The original task notes asks for negative p to be rejected through all three solvers.
The frozen `check_parameter` rejects `p < 0` only for solver 11. The port
follows that source rule: solver 11 returns the source text and a correction;
solvers 12 and 13 accept finite negative p. Unit tests check all three paths.
The original task notes also says the corpus misses dual branches. Counts below show that
it reaches all three solver 13 shrinking sites and both dual limit exits.
The zero counts at solver 12's bound shrinking sites follow its infinite
bound; no finite beta can reach that bound.

All 13 solver 11 cases, 12 solver 12 cases and 12 solver 13 cases match all
saved fields by bits, worst absolute error 0, worst relative error 0 and
worst 0 ULP. Reports are `reference/parity/solver11.json`, `solver12.json`
and `solver13.json` in that folder. Cases cover C of 1e-6 and 1e6, eps of
1e-1 and 1e-8, bias 1 and -1, sparse and dense rows, copied rows, one row,
zero features and free bias for solver 11. Regression targets stay as f64.
Tests also check empty inputs, nonfinite targets and p, initial weight
sizes and values, class weights, and the solver 21 initial-weight error.

`train_with_svr_threshold` is a hidden check entry point. Its bool argument
selects a test path that halves p only in the dual solver. The regular
train path always uses the input p. The normal test requires this mutation
to fail for both dual solvers. The ignored test shows the solver 12 failure
on `s12_C_large`, at `decision_values[0][0]`: actual bits
`0x3ff3d60ddaad4f12`, reference bits `0x3ff3d60484c91768`.

### Extra SVR C problems and branch counts

`tests/fixtures/svr_reference.cpp` uses the frozen reader and writer with
the unchanged vendored solver. It prints doubles with `%.17g` and starts
seed 1 for each train call. The Python script next to it builds and runs
both C++ programs without network access. It makes six inputs with Python
seed 1113: equal targets; one outlier; p zero; p 100; a copied row with
C 1e-3; and a zero column with C 1e3 and eps 1e-30. Each has 12 rows, five
features and bias 1 with its penalty. All six problems match all saved
fields by bits for solvers 11, 12 and 13. Their O0 coverage and O2 outputs
are equal by bytes.

`svr_extra.json` stores each input hash and its C results.
`svr_extra.build.json` stores the compiler, build commands, hashes of all
compiled sources and the included driver, the aggregate output hash,
each raw output hash, and GCC branch counts. `{repo}` in a command means
the absolute repository path. Tests check source, input and aggregate
output hashes before comparing the values. The fixture README gives the
command to make the inputs, outputs and coverage counts again.

These are execution counts from GCC coverage on the unchanged C source.
Each count is one branch entry. Corpus runs are separate for each solver.
Extra counts are totals over the six problems.

| Source path | Line | Corpus 12 | Extra 12 | Corpus 13 | Extra 13 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Remove a row with beta zero | 1200 | 178 | 2 | 394 | 7 |
| Remove a row at the upper bound | 1212 | 0 | 0 | 189 | 5 |
| Remove a row at the lower bound | 1224 | 0 | 0 | 165 | 6 |
| Reset all active rows | 1269 | 4 | 1 | 13 | 3 |
| Exit at 300 steps, return dual weights | 2742 | 0 | 0 | 1 | 1 |
| Exit at 300 steps, switch to Newton | 2751 | 1 | 1 | 0 | 0 |

Solver 11 has zero entries at these dual sites. The corpus misses none of
the measured paths that apply to each dual solver. These counts measure
entries, not every outcome of each comparison. Both extra limit exits
come from the zero-column C 1e3 case. This checks the source behavior at
the limit; it does not claim convergence. No frozen reference file or
tolerance was changed. No speed claim is made.

## Solver 21: source map and measured parity

`src/oneclass_svm.rs` ports `solve_oneclass_svm`, `partition`,
`quick_select_min_k` and `compare_feature_node` from the frozen source.
The initial alpha vector has `floor(nu*l)` ones, then the remaining
fraction, then zeros. The fit keeps input row order. It uses the same
active set, pair selection, bound updates, 1e-12 pair test, curvature
floor and 1000-step limit. Sparse norms, dots, products and axpy updates
keep the feature order. The objective and support count only feed the
C print callback; the Rust solver omits these print-only sums.

The original task notes calls for a random permutation. The source has no shuffle in
this solver. It uses glibc rand for the quick-select pivots. The Rust
port keeps each pivot call and its order. It replaces recursive selection
with a loop. The selected nodes sort by value, then by row index; this
full order has no equal distinct nodes, so the Rust sort gives the same
order as C qsort. Training starts the glibc stream at seed 1.

The source computes rho as the mean gradient of free alpha values. If
none are free, it uses `(ub + lb)/2`, in that order. Prediction subtracts
rho and returns +1 only for a positive score; zero returns -1. Row labels
are ignored and may be finite nonintegers. C, p and class weights do not
change the fit. C must still pass the common positive-value check. Bias
is rejected with the frozen C text before the free-bias check.

The original task notes asks for the C text for invalid nu. LIBLINEAR 2.50 has no nu
range check or error text. The Rust range check requires `0 < nu <= 1`
and says `Set nu to a number greater than zero and at most 1.` This is
an added input check. At nu 1, all alpha values equal 1. The source rho
formula then gives positive infinity; Rust keeps that result. The frozen
JSON writer rejects nonfinite outputs, so nu 1 has no finite parity claim.
A Rust test checks this source boundary. No tolerance is changed.

The 11 frozen cases all use nu 0.5; the corpus does not contain the nu
extremes named in the task notes. All 11 cases match every field by bits, with
worst absolute and relative error 0 and worst 0 ULP. The tests require the
unchanged `reference/tolerances.json` and write
[reference/parity/solver21.json](reference/parity/solver21.json).
Six extra C problems in `tests/fixtures/oneclass_extra.json` also match
every field by bits. They cover nu 0.01, 0.5 and 0.99, equal rows, a copied
row, a zero column, one row and features scaled by 1e3. The scaled case
uses eps 1e-16 to reach the iteration limit. The equal-row case checks the
rho bound formula. These cases have fractional labels to check that labels
are ignored. A hand-made model checks positive, zero and negative scores.
The parameter tests check bias, nu, empty input and unknown solver 99.

`tests/fixtures/oneclass_reference.cpp` uses the frozen case reader and
writer. The build record next to the extra JSON stores the source hashes,
compiler, build commands and SHA-256 of the output. Replace `{repo}` with
the absolute repo path and `{build}` with an absolute build folder. Build
and run one command at a time with `RAYON_NUM_THREADS=2 CARGO_BUILD_JOBS=2`.
The O0 build uses GCC coverage counters in the unchanged source. For each
input, delete its gcda files, run the program, then use the saved gcov
command. The metadata keeps the per-case branch-body counts. All six O0
and O2 outputs are equal by bytes. All 11 corpus O0 outputs also equal the
frozen O2 bytes.

Counts below name executed branch bodies in `linear.cpp`, not evaluations
of their conditions. Line 2327 removes one row from the active set; line
2315 restores all rows. Line 2437 reports the 1000-step limit. Lines 2470
and 2472 select the two rho formulas.

| Input group | Shrink, 2327 | Restore, 2315 | Limit, 2437 | Free rho, 2470 | Bound rho, 2472 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Frozen corpus, 11 cases | 1104 | 13 | 0 | 11 | 0 |
| Extra nu_half | 0 | 0 | 0 | 0 | 1 |
| Extra nu_high_duplicate | 4 | 1 | 0 | 1 | 0 |
| Extra nu_low | 0 | 0 | 0 | 1 | 0 |
| Extra one_row | 0 | 0 | 0 | 1 | 0 |
| Extra scaled | 0 | 0 | 1 | 1 | 0 |
| Extra zero_column | 0 | 0 | 0 | 1 | 0 |
| Extra total, 6 problems | 4 | 1 | 1 | 5 | 1 |

The corpus misses the iteration-limit exit and the rho bound formula.
The six extra problems cover both. The ignored `changed_rho_matches_reference`
test changes rho by 1 in the same solver path. It must fail the same field
comparison. The normal test also checks that this mutation is rejected.
No speed claim is made.

## Hand-run checks

Run the heavy checks by hand. Fuzzing feeds bytes to the case reader, to training and to prediction. Proofs check the weight index bounds, the random stream indices and one feature index rule. Miri runs the determinism test and the random and BLAS tests. Mutation checks that the tests reject a changed solver. These checks do not run on every change.
