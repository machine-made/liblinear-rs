// SPDX-License-Identifier: BSD-3-Clause
#![forbid(unsafe_code)]

//! Safe LIBLINEAR types, a case reader and all 12 LIBLINEAR solvers.
//!
//! # Contract
//!
//! Class: production kernel
//!
//! ## Purpose
//!
//! The crate reads a case file and trains all 12 LIBLINEAR 2.50 solvers.
//! `train` returns an owned model and does not change the input rows.
//! `CONTRACT.md` holds the units, the shapes and the parameter errors.
//!
//! ## Invariants
//!
//! Feature indices start at 1 and increase in each row.
//! Weights use feature order, then class order.
//! Each train call starts the random stream at seed 1.
//! Sums keep the C operation order and do not use a fused multiply-add.
//! Saved reference cases match by bits.
//!
//! ## Error model
//!
//! `Error` names the failed check and the correction.
//! The library does not print.
//! An empty problem returns an error and has no numeric result.
//!
//! ## Determinism class
//!
//! Results are bit-identical across runs and thread counts on one CPU type.
//! No code path uses threads.
//!
//! ## Unsafe leaves
//!
//! The crate has no unsafe code.
//!
//! ## Features
//!
//! The crate has no optional feature and no C backend.
//!
//! ## What is not proven
//!
//! Equal saved builds do not by themselves prove the port.
//! The solver checks measure that match only on the saved cases.
//! Bit equality on those cases does not prove other inputs.
//! The exp and log table does not prove another C library.
//! Nonfinite inputs have no solver result.
//! Nu of 1 has no finite one-class parity result.
//! A NaN from the extra BLAS checks can change its sign or payload.
//! These checks do not prove a speed claim.
//!
//! ```
//! use liblinear::{Feature, Parameter, Problem, SolverType, train};
//! let parameter = Parameter {
//!     solver_type: SolverType::L2rL2lossSvc,
//!     eps: 0.1,
//!     c: 1.0,
//!     class_weights: Vec::new(),
//!     p: 0.1,
//!     nu: 0.5,
//!     init_sol: None,
//!     regularize_bias: true,
//!     w_recalc: false,
//! };
//! let problem = Problem {
//!     solvers: vec![SolverType::L2rL2lossSvc],
//!     parameter: parameter.clone(),
//!     nr_feature: 1,
//!     bias: -1.0,
//!     labels: vec![1.0],
//!     rows: vec![vec![Feature { index: 1, value: 1.0 }]],
//! };
//! let model = train(&problem, &parameter)?;
//! assert_eq!(model.predict(&problem)?.len(), 1);
//! # Ok::<(), liblinear::Error>(())
//! ```

pub(crate) mod blas;
mod dual_l1l2_svc;
mod dual_lr;
mod dual_svr;
pub mod error;
mod l1r_l2_svc;
mod l1r_lr;
mod l2r_l2_svc;
mod l2r_l2_svr;
mod l2r_lr;
mod mcsvm_cs;
pub mod model;
mod oneclass_svm;
pub mod params;
pub mod problem;
pub(crate) mod rand;
pub mod train;
mod tron;

pub use error::Error;
pub use model::Model;
pub use params::{Parameter, SolverType};
pub use problem::{Feature, Problem, read_problem};
pub use train::train;

#[cfg(test)]
pub(crate) use train::{
    train_with_dot, train_with_log, train_with_mcsvm_sort, train_with_oneclass_rho,
    train_with_rand, train_with_shrinking, train_with_svr_threshold,
};

#[cfg(test)]
#[expect(
    clippy::unwrap_used,
    reason = "Tests read known files and saved results"
)]
#[expect(
    clippy::arithmetic_side_effects,
    reason = "Tests count JSON fields and compared values"
)]
#[path = "../tests/common/mod.rs"]
mod parity_common;

#[cfg(test)]
#[expect(
    clippy::unwrap_used,
    reason = "Tests read known files and saved results"
)]
#[path = "../tests/blas.rs"]
mod blas_checks;

#[cfg(test)]
#[expect(
    clippy::indexing_slicing,
    reason = "Tests read known JSON fields and saved rows"
)]
#[expect(
    clippy::unwrap_used,
    reason = "Tests read known files and saved results"
)]
#[path = "../tests/rand.rs"]
mod rand_checks;

#[cfg(test)]
#[expect(
    clippy::indexing_slicing,
    reason = "Tests read known JSON fields and saved rows"
)]
#[expect(
    clippy::panic,
    reason = "A parity mismatch prints the failed comparison"
)]
#[expect(
    clippy::unwrap_used,
    reason = "Tests read known files and saved results"
)]
#[expect(
    clippy::arithmetic_side_effects,
    reason = "Tests count JSON fields and compared values"
)]
#[expect(
    clippy::expect_used,
    reason = "A failed parity check must return the error text"
)]
#[path = "../tests/solver2.rs"]
mod solver2_checks;

#[cfg(test)]
#[expect(
    clippy::indexing_slicing,
    reason = "Tests read known JSON fields and saved rows"
)]
#[expect(
    clippy::panic,
    reason = "A parity mismatch prints the failed comparison"
)]
#[expect(
    clippy::unwrap_used,
    reason = "Tests read known files and saved results"
)]
#[expect(
    clippy::expect_used,
    reason = "A failed parity check must return the error text"
)]
#[path = "../tests/solver4.rs"]
mod solver4_checks;

#[cfg(test)]
#[expect(
    clippy::indexing_slicing,
    reason = "Tests read known JSON fields and saved rows"
)]
#[expect(
    clippy::panic,
    reason = "A parity mismatch prints the failed comparison"
)]
#[expect(
    clippy::unwrap_used,
    reason = "Tests read known files and saved results"
)]
#[expect(
    clippy::expect_used,
    reason = "A failed parity check must return the error text"
)]
#[path = "../tests/solver07.rs"]
mod solver07_checks;

#[cfg(test)]
#[expect(
    clippy::indexing_slicing,
    reason = "Tests read known JSON fields and saved rows"
)]
#[expect(
    clippy::panic,
    reason = "A parity mismatch prints the failed comparison"
)]
#[expect(
    clippy::unwrap_used,
    reason = "Tests read known files and saved results"
)]
#[expect(
    clippy::expect_used,
    reason = "A failed parity check must return the error text"
)]
#[path = "../tests/solver11_13.rs"]
mod solver11_checks;

#[cfg(test)]
#[expect(
    clippy::indexing_slicing,
    reason = "Tests read known JSON fields and saved rows"
)]
#[expect(
    clippy::panic,
    reason = "A parity mismatch prints the failed comparison"
)]
#[expect(
    clippy::unwrap_used,
    reason = "Tests read known files and saved results"
)]
#[expect(
    clippy::expect_used,
    reason = "A failed parity check must return the error text"
)]
#[path = "../tests/solver13.rs"]
mod solver13_checks;

#[cfg(test)]
#[expect(
    clippy::indexing_slicing,
    reason = "Tests read known JSON fields and saved rows"
)]
#[expect(
    clippy::panic,
    reason = "A parity mismatch prints the failed comparison"
)]
#[expect(
    clippy::unwrap_used,
    reason = "Tests read known files and saved results"
)]
#[expect(
    clippy::expect_used,
    reason = "A failed parity check must return the error text"
)]
#[path = "../tests/solver21.rs"]
mod solver21_checks;

#[cfg(test)]
#[expect(
    clippy::indexing_slicing,
    reason = "Tests read known JSON fields and saved rows"
)]
#[expect(
    clippy::panic,
    reason = "A parity mismatch prints the failed comparison"
)]
#[expect(
    clippy::unwrap_used,
    reason = "Tests read known files and saved results"
)]
#[expect(
    clippy::expect_used,
    reason = "A failed parity check must return the error text"
)]
#[path = "../tests/solver56.rs"]
mod solver56_checks;
