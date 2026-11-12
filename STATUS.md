# Status

The LIBLINEAR 2.50 reference is frozen. The Rust crate has a contract for all
12 solvers, exact comparison rules, input and output types, and a case file
reader. All 12 solvers can train and predict.
Solvers 0 through 7 match 14, 14, 15, 13, 14, 14, 14 and 13 frozen cases.
Solvers 11, 12 and 13 match 13, 12 and 12 frozen cases. Solver 21 matches
all 11 frozen cases, including rho. All fields are
bit equal, with worst 0 ULP.

## Checks

- Solver 21 matches six extra C problems with nu 0.01, 0.5 and 0.99.
- Its extra cases cover equal and copied rows, a zero column, one row and scaled features.
- The extra scaled case reaches the 1000-step limit; the equal-row case checks bound rho.
- GCC counters record shrinking, restore, iteration-limit and both rho branches.
- A changed rho update fails the same field comparison.
- One-class prediction subtracts rho and returns -1 for a score of zero.

- Six extra C problems match every field for solvers 11, 12 and 13 by bits.
- SVR cases check equal targets, one outlier, p zero and large p, copied rows and a zero column.
- GCC counts check all three dual SVR shrinking sites and both 300-step exits.
- The changed dual SVR epsilon threshold fails parity for solvers 12 and 13.
- Negative p is rejected only for solver 11, as in the frozen source.

- The reference has 158 cases and 316 result files from `-O0` and `-O2`.
- The Rust tests check every case against its row and feature header counts.
- The golden loader checks values, floating-point bits, file bytes and shapes.
- Each output field, including null rho and labels, must have a comparison rule.
- Invalid labels, indices, headers and missing files return error messages.
- A one-row case returns one row from the reader.
- Solvers 0 through 7 match weights, scores, predictions, labels and all other saved fields.
- Solver 4 matches six extra multiclass C problems, including its iteration limit.
- Its Rust sort matches 10000 C qsort arrays by bits, with ties and signed zeros.
- Its changed-sort mutation fails parity. C branch counts are in the contract.
- Six extra L1 C problems match all fields for solvers 5 and 6 by bits.
- C coverage checks both L1 line-search exits and all shrinking comparisons.
- The changed-shrinking mutation fails parity for both L1 solvers.
- The libm table matches 2000 exp and 2000 log outputs by bits on glibc 2.39.
- Extra logistic C cases check multiclass fits, free bias and initial weights.
- The extra solver 7 case covers two Newton fallbacks after 300 dual steps.
- The changed-logarithm mutation fails the comparison for both logistic solvers.
- The safe glibc stream matches 10000 C outputs for each of seeds 1 and 2.
- Extra dual C cases check multiclass stream use and weight rebuilds.
- The changed-generator mutation fails the same model comparison.
- The analytic solver 1 case covers the Newton fallback after 300 dual steps.
- The analytic weight is bit equal to `4000/4001`.
- The reversed-dot mutation fails the same comparison.
- The C BLAS table checks loop boundaries, scales, subnormal values and signed zeros.
- Extra C cases check multiclass weights, free bias and binary label order.
- Initial weights, prediction ties and train input errors are checked.
- CI checks format, lint and tests on pull requests and manual runs.

The contract is in [CONTRACT.md](CONTRACT.md). The build record is in
[reference/BUILD.md](reference/BUILD.md). The measured build spread is in
[reference/spread.json](reference/spread.json). All tolerance rules start
from that zero spread and require a new measurement before a change.

## Next

Complete the parity adapter. The project licence is still open.

## Limits

The saved runs use Linux, glibc and g++ 13.3.0. Dual solvers use the C library
random number generator. A change in that library can change the saved
results. Equal results from two C++ builds do not prove a Rust solver
matches. Parity is measured in the reports for
[solver 0](reference/parity/solver0.json), [solver 1](reference/parity/solver1.json), [solver 2](reference/parity/solver2.json),
[solver 3](reference/parity/solver3.json), [solver 4](reference/parity/solver4.json), [solver 5](reference/parity/solver5.json),
[solver 6](reference/parity/solver6.json), [solver 7](reference/parity/solver7.json),
[solver 11](reference/parity/solver11.json), [solver 12](reference/parity/solver12.json)
[solver 13](reference/parity/solver13.json) and
[solver 21](reference/parity/solver21.json). The frozen corpus leaves weight
rebuilds off; extra C cases check that branch. The corpus uses finite values.
No speed claim is made.
