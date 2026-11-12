# Changelog

## Unreleased

- Added fuzz targets for the case reader, training and prediction, with a seed corpus
- Added proof harnesses for the weight index bounds, the random stream indices and one feature index rule
- Added a workflow that runs fuzz, proof, Miri and mutation checks by hand
- Added a crate contract for purpose, invariants, errors, determinism, unsafe code, features and unproven claims
- Added bit comparison of train and predict for solvers 0, 1, 2, 4, 5, 11 and 21 on one thread and on four threads
- Added a compiling example on each public type, call, field and variant
- Changed solver numbers so a new variant does not break an existing match
- Changed the BLAS loops, the random stream and the parity train hooks to crate-private items
- Improved the L1, multiclass and one-class solvers so they return index and overflow errors
- Improved the squared-hinge Newton step by checking each active row once
- Improved the dual and L2 solvers so they return index and overflow errors
- Improved the squared-hinge score by checking the weight length once
- Changed bounds checks in support code to return index and overflow errors
- Added train and predict timings for solvers 0, 1, 2, 5 and 11 on one fixed problem
- Added typed errors for unequal slices, oversized counts, missing rho and missing labels
- Added one-class solver 21 with the frozen pair selection, shrinking and rho rules
- Added exact-bit proof for all 11 solver 21 cases and six extra C problems
- Added C counts for one-class shrinking, the iteration limit and both rho formulas
- Added a changed rho update check that fails parity
- Added support vector regression solvers 11, 12 and 13 with the frozen Newton and dual steps
- Added bit equality reports for 13 solver 11 cases and 12 cases each for solvers 12 and 13
- Added six extra C regression problems and counts for shrinking and both dual iteration limits
- Added a changed epsilon threshold check that fails parity for both dual regression solvers
- Added solver 4 with the frozen joint multiclass steps and feature-major weights
- Added bit equality proof for all 14 solver 4 cases and six extra C problems
- Added a C qsort table for 10000 seeded arrays and a sort mutation check
- Added C coverage counts for multiclass shrinking and the iteration-limit exit
- Added L1 solvers 5 and 6 with the frozen coordinate, shrinking and line-search rules
- Added bit equality reports for all 14 solver 5 cases and all 14 solver 6 cases
- Added six extra C problems and coverage counts for both L1 line-search exits
- Added a changed-shrinking check that fails parity for both L1 solvers
- Added logistic regression solvers 0 and 7 with the frozen Newton and dual step order
- Added a C libm table and bit checks for 2000 exp and 2000 log outputs
- Added bit equality reports for all 14 solver 0 cases and all 13 solver 7 cases
- Added C checks for logistic multiclass fits, initial weights, free bias and the Newton fallback
- Added a changed-logarithm check that fails the model comparison
- Added dual SVC solvers 1 and 3 with the frozen shuffle, shrinking and stop tests
- Added a safe glibc random stream and checks for 10000 C outputs from each of two seeds
- Added bit equality reports for all 14 solver 1 cases and all 13 solver 3 cases
- Added C checks for dual multiclass fits and weight rebuilds, and a changed-generator mutation check
- Added solver 2 training and prediction with the frozen Newton and BLAS operation order
- Added bit equality checks for all 15 solver 2 cases and a saved parity report
- Added a reversed-sum check that fails the same model comparison
- Added a C BLAS table and checks for class order, multiclass weights, free bias and initial weights
- Added a contract for all 12 solvers, input shapes, output shapes and parameter errors
- Added exact comparison rules from the measured build spread for every saved output field
- Added Rust types and a case file reader with input error messages
- Added checks for all 158 input cases and all 316 saved result files
- Added Rust format, lint and test checks for pull requests and manual runs
- Changed scoring to return an error when rho or class labels are absent
- Changed integration checks to deny warnings, test both feature sets and build the docs
- Changed one-class prediction to subtract rho and return -1 for a zero score
- Changed unknown solver errors to say that the solver number does not exist
- Changed regression training to keep target order and use one weight vector
- Changed solver errors to list solvers 0 through 7 and 11 through 13
- Changed solver errors to list all solvers from 0 through 7
- Changed solver errors to list solvers 0, 1, 2, 3, 5, 6 and 7
- Changed solver dispatch to keep the solver 1 Newton fallback and list all ported solvers in errors

## 0.0.1

- Added unchanged LIBLINEAR 2.50 source with source hashes and upstream copyright terms
- Added 158 fixed input cases for all 12 solvers and saved results from two compiler builds
- Added the reference driver, build script, hash checks, rerun checks and build spread
- Added an analytic weight check and tests for a changed result byte and a missing solver
