// SPDX-License-Identifier: BSD-3-Clause
//! Solver numbers and training inputs.
//!
//! ```
//! use liblinear::{Parameter, SolverType};
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
//! assert_eq!(parameter.solver_type as i32, 2);
//! ```

use crate::Error;

/// Solver numbers from LIBLINEAR 2.50.
///
/// ```
/// use liblinear::SolverType;
/// assert_eq!(SolverType::try_from(2)? as i32, 2);
/// # Ok::<(), liblinear::Error>(())
/// ```
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(i32)]
#[non_exhaustive]
pub enum SolverType {
    /// L2-regularized logistic regression.
    ///
    /// ```
    /// assert_eq!(liblinear::SolverType::L2rLr as i32, 0);
    /// ```
    L2rLr = 0,
    /// L2-regularized L2-loss SVC, dual form.
    ///
    /// ```
    /// assert_eq!(liblinear::SolverType::L2rL2lossSvcDual as i32, 1);
    /// ```
    L2rL2lossSvcDual = 1,
    /// L2-regularized L2-loss SVC, primal form.
    ///
    /// ```
    /// assert_eq!(liblinear::SolverType::L2rL2lossSvc as i32, 2);
    /// ```
    L2rL2lossSvc = 2,
    /// L2-regularized L1-loss SVC, dual form.
    ///
    /// ```
    /// assert_eq!(liblinear::SolverType::L2rL1lossSvcDual as i32, 3);
    /// ```
    L2rL1lossSvcDual = 3,
    /// Joint multiclass SVM.
    ///
    /// ```
    /// assert_eq!(liblinear::SolverType::McsvmCs as i32, 4);
    /// ```
    McsvmCs = 4,
    /// L1-regularized L2-loss SVC.
    ///
    /// ```
    /// assert_eq!(liblinear::SolverType::L1rL2lossSvc as i32, 5);
    /// ```
    L1rL2lossSvc = 5,
    /// L1-regularized logistic regression.
    ///
    /// ```
    /// assert_eq!(liblinear::SolverType::L1rLr as i32, 6);
    /// ```
    L1rLr = 6,
    /// L2-regularized logistic regression, dual form.
    ///
    /// ```
    /// assert_eq!(liblinear::SolverType::L2rLrDual as i32, 7);
    /// ```
    L2rLrDual = 7,
    /// L2-regularized L2-loss SVR, primal form.
    ///
    /// ```
    /// assert_eq!(liblinear::SolverType::L2rL2lossSvr as i32, 11);
    /// ```
    L2rL2lossSvr = 11,
    /// L2-regularized L2-loss SVR, dual form.
    ///
    /// ```
    /// assert_eq!(liblinear::SolverType::L2rL2lossSvrDual as i32, 12);
    /// ```
    L2rL2lossSvrDual = 12,
    /// L2-regularized L1-loss SVR, dual form.
    ///
    /// ```
    /// assert_eq!(liblinear::SolverType::L2rL1lossSvrDual as i32, 13);
    /// ```
    L2rL1lossSvrDual = 13,
    /// One-class SVM.
    ///
    /// ```
    /// assert_eq!(liblinear::SolverType::OneclassSvm as i32, 21);
    /// ```
    OneclassSvm = 21,
}

impl TryFrom<i32> for SolverType {
    type Error = Error;

    fn try_from(value: i32) -> Result<Self, Error> {
        match value {
            0 => Ok(Self::L2rLr),
            1 => Ok(Self::L2rL2lossSvcDual),
            2 => Ok(Self::L2rL2lossSvc),
            3 => Ok(Self::L2rL1lossSvcDual),
            4 => Ok(Self::McsvmCs),
            5 => Ok(Self::L1rL2lossSvc),
            6 => Ok(Self::L1rLr),
            7 => Ok(Self::L2rLrDual),
            11 => Ok(Self::L2rL2lossSvr),
            12 => Ok(Self::L2rL2lossSvrDual),
            13 => Ok(Self::L2rL1lossSvrDual),
            21 => Ok(Self::OneclassSvm),
            _ => Err(Error::UnknownSolver(value)),
        }
    }
}

/// Training inputs. See `CONTRACT.md` for units and valid ranges.
///
/// ```
/// use liblinear::{Parameter, SolverType};
/// let parameter = Parameter {
///     solver_type: SolverType::L2rL2lossSvc,
///     eps: 0.1,
///     c: 1.0,
///     class_weights: Vec::new(),
///     p: 0.1,
///     nu: 0.5,
///     init_sol: None,
///     regularize_bias: true,
///     w_recalc: false,
/// };
/// assert_eq!(parameter.c.to_bits(), 1.0_f64.to_bits());
/// ```
#[derive(Clone, Debug, PartialEq)]
pub struct Parameter {
    /// Solver number for this call.
    ///
    /// ```
    /// # use liblinear::{Parameter, SolverType};
    /// # let parameter = Parameter { solver_type: SolverType::L2rLr, eps: 0.1, c: 1.0, class_weights: Vec::new(), p: 0.1, nu: 0.5, init_sol: None, regularize_bias: true, w_recalc: false };
    /// assert_eq!(parameter.solver_type as i32, 0);
    /// ```
    pub solver_type: SolverType,
    /// Positive stop limit. The selected solver uses its own stop test.
    ///
    /// ```
    /// # use liblinear::{Parameter, SolverType};
    /// # let parameter = Parameter { solver_type: SolverType::L2rLr, eps: 0.1, c: 1.0, class_weights: Vec::new(), p: 0.1, nu: 0.5, init_sol: None, regularize_bias: true, w_recalc: false };
    /// assert_eq!(parameter.eps.to_bits(), 0.1_f64.to_bits());
    /// ```
    pub eps: f64,
    /// The C value in the case file.
    ///
    /// ```
    /// # use liblinear::{Parameter, SolverType};
    /// # let parameter = Parameter { solver_type: SolverType::L2rLr, eps: 0.1, c: 1.0, class_weights: Vec::new(), p: 0.1, nu: 0.5, init_sol: None, regularize_bias: true, w_recalc: false };
    /// assert_eq!(parameter.c.to_bits(), 1.0_f64.to_bits());
    /// ```
    pub c: f64,
    /// Class label and cost multiplier, in input order.
    ///
    /// ```
    /// # use liblinear::{Parameter, SolverType};
    /// # let parameter = Parameter { solver_type: SolverType::L2rLr, eps: 0.1, c: 1.0, class_weights: vec![(1, 2.0)], p: 0.1, nu: 0.5, init_sol: None, regularize_bias: true, w_recalc: false };
    /// assert_eq!(parameter.class_weights, vec![(1, 2.0)]);
    /// ```
    pub class_weights: Vec<(i32, f64)>,
    /// Loss width for solvers 11, 12 and 13, in target units.
    ///
    /// ```
    /// # use liblinear::{Parameter, SolverType};
    /// # let parameter = Parameter { solver_type: SolverType::L2rL2lossSvr, eps: 0.1, c: 1.0, class_weights: Vec::new(), p: 0.1, nu: 0.5, init_sol: None, regularize_bias: true, w_recalc: false };
    /// assert_eq!(parameter.p.to_bits(), 0.1_f64.to_bits());
    /// ```
    pub p: f64,
    /// One-class fraction for solver 21. The value is greater than zero and at most 1.
    ///
    /// ```
    /// # use liblinear::{Parameter, SolverType};
    /// # let parameter = Parameter { solver_type: SolverType::OneclassSvm, eps: 0.1, c: 1.0, class_weights: Vec::new(), p: 0.1, nu: 0.5, init_sol: None, regularize_bias: true, w_recalc: false };
    /// assert_eq!(parameter.nu.to_bits(), 0.5_f64.to_bits());
    /// ```
    pub nu: f64,
    /// Initial weights for solvers 0, 2 and 11. `None` starts from zero.
    ///
    /// ```
    /// # use liblinear::{Parameter, SolverType};
    /// # let parameter = Parameter { solver_type: SolverType::L2rLr, eps: 0.1, c: 1.0, class_weights: Vec::new(), p: 0.1, nu: 0.5, init_sol: None, regularize_bias: true, w_recalc: false };
    /// assert!(parameter.init_sol.is_none());
    /// ```
    pub init_sol: Option<Vec<f64>>,
    /// `true` applies the bias penalty.
    ///
    /// ```
    /// # use liblinear::{Parameter, SolverType};
    /// # let parameter = Parameter { solver_type: SolverType::L2rLr, eps: 0.1, c: 1.0, class_weights: Vec::new(), p: 0.1, nu: 0.5, init_sol: None, regularize_bias: true, w_recalc: false };
    /// assert!(parameter.regularize_bias);
    /// ```
    pub regularize_bias: bool,
    /// `true` rebuilds weights at the end of solver 1 or 3.
    ///
    /// ```
    /// # use liblinear::{Parameter, SolverType};
    /// # let parameter = Parameter { solver_type: SolverType::L2rL2lossSvcDual, eps: 0.1, c: 1.0, class_weights: Vec::new(), p: 0.1, nu: 0.5, init_sol: None, regularize_bias: true, w_recalc: false };
    /// assert!(!parameter.w_recalc);
    /// ```
    pub w_recalc: bool,
}
