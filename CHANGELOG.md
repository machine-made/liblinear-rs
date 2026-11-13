# Changelog

## Unreleased

- Added a comparison of training time, prediction time and peak memory with LIBLINEAR 2.50 on 20 real dataset and solver pairs

## 0.0.1

- Added a pure Rust port of LIBLINEAR 2.50 with no C code, no unsafe code and no optional feature
- Added the 12 solvers 0 to 7, 11 to 13 and 21: logistic regression, support vector classification, multiclass classification, support vector regression and one-class detection
- Added the types `Problem`, `Parameter`, `Model`, `Feature`, `SolverType` and `Error`, the `train` call, `Model::predict` and `Model::decision_values`, and `read_problem` for case files
- Added errors that name the failed check and the correction, and index and overflow checks that return an error instead of a panic
- Added exact-bit results against the unchanged LIBLINEAR 2.50 source on 158 saved cases, built with g++ at `-O0` and `-O2`, and a script that checks the saved reference
- Added checks that the `exp` and `log` results, the random stream, the sort order and the BLAS sums match the C library by bits
- Added bit-identical results across runs and thread counts on one CPU type
- Added mutation checks that fail when a log, a sum order, the random generator, a sort, a shrinking rule or a threshold changes
- Added fuzz targets for the case reader, training and prediction, proof harnesses for index bounds, and Miri runs, started by hand in one workflow
- Added `CONTRACT.md` with units, shapes, error rules and a source map for every solver, and `docs/PARITY.md` with the result for each solver
- Added train and predict timings for five solvers; the crate makes no speed claim
