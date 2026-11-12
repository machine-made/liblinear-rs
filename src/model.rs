// SPDX-License-Identifier: BSD-3-Clause
//! Fitted model, scores and predictions.
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
//! assert_eq!(model.nr_feature, 1);
//! # Ok::<(), liblinear::Error>(())
//! ```

use crate::{Error, Parameter, Problem, SolverType};

/// A fitted model and its outputs for the input rows.
///
/// Weights use feature order, then class order. The last feature holds the
/// bias weight when bias is nonnegative. See `CONTRACT.md` for array sizes.
///
/// ```
/// use liblinear::{Feature, Parameter, Problem, SolverType, train};
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
/// let problem = Problem {
///     solvers: vec![SolverType::L2rL2lossSvc],
///     parameter: parameter.clone(),
///     nr_feature: 1,
///     bias: -1.0,
///     labels: vec![1.0],
///     rows: vec![vec![Feature { index: 1, value: 1.0 }]],
/// };
/// let model = train(&problem, &parameter)?;
/// assert_eq!(model.weights.len(), 1);
/// # Ok::<(), liblinear::Error>(())
/// ```
#[derive(Clone, Debug, PartialEq)]
pub struct Model {
    /// Holds the solver number, C and eps saved in each result.
    ///
    /// ```
    /// # use liblinear::{Feature, Parameter, Problem, SolverType, train};
    /// # let parameter = Parameter { solver_type: SolverType::L2rL2lossSvc, eps: 0.1, c: 1.0, class_weights: Vec::new(), p: 0.1, nu: 0.5, init_sol: None, regularize_bias: true, w_recalc: false };
    /// # let problem = Problem { solvers: vec![SolverType::L2rL2lossSvc], parameter: parameter.clone(), nr_feature: 1, bias: -1.0, labels: vec![1.0], rows: vec![vec![Feature { index: 1, value: 1.0 }]] };
    /// # let model = train(&problem, &parameter)?;
    /// assert_eq!(model.parameter.c.to_bits(), 1.0_f64.to_bits());
    /// # Ok::<(), liblinear::Error>(())
    /// ```
    pub parameter: Parameter,
    /// Class count. Regression and one-class models use 2.
    ///
    /// ```
    /// # use liblinear::{Feature, Parameter, Problem, SolverType, train};
    /// # let parameter = Parameter { solver_type: SolverType::L2rL2lossSvc, eps: 0.1, c: 1.0, class_weights: Vec::new(), p: 0.1, nu: 0.5, init_sol: None, regularize_bias: true, w_recalc: false };
    /// # let problem = Problem { solvers: vec![SolverType::L2rL2lossSvc], parameter: parameter.clone(), nr_feature: 1, bias: -1.0, labels: vec![1.0], rows: vec![vec![Feature { index: 1, value: 1.0 }]] };
    /// # let model = train(&problem, &parameter)?;
    /// assert_eq!(model.nr_class, 1);
    /// # Ok::<(), liblinear::Error>(())
    /// ```
    pub nr_class: usize,
    /// Feature count before the bias feature.
    ///
    /// ```
    /// # use liblinear::{Feature, Parameter, Problem, SolverType, train};
    /// # let parameter = Parameter { solver_type: SolverType::L2rL2lossSvc, eps: 0.1, c: 1.0, class_weights: Vec::new(), p: 0.1, nu: 0.5, init_sol: None, regularize_bias: true, w_recalc: false };
    /// # let problem = Problem { solvers: vec![SolverType::L2rL2lossSvc], parameter: parameter.clone(), nr_feature: 1, bias: -1.0, labels: vec![1.0], rows: vec![vec![Feature { index: 1, value: 1.0 }]] };
    /// # let model = train(&problem, &parameter)?;
    /// assert_eq!(model.nr_feature, 1);
    /// # Ok::<(), liblinear::Error>(())
    /// ```
    pub nr_feature: usize,
    /// Copy of the input bias.
    ///
    /// ```
    /// # use liblinear::{Feature, Parameter, Problem, SolverType, train};
    /// # let parameter = Parameter { solver_type: SolverType::L2rL2lossSvc, eps: 0.1, c: 1.0, class_weights: Vec::new(), p: 0.1, nu: 0.5, init_sol: None, regularize_bias: true, w_recalc: false };
    /// # let problem = Problem { solvers: vec![SolverType::L2rL2lossSvc], parameter: parameter.clone(), nr_feature: 1, bias: -1.0, labels: vec![1.0], rows: vec![vec![Feature { index: 1, value: 1.0 }]] };
    /// # let model = train(&problem, &parameter)?;
    /// assert!(model.bias < 0.0);
    /// # Ok::<(), liblinear::Error>(())
    /// ```
    pub bias: f64,
    /// Set only for solver 21.
    ///
    /// ```
    /// # use liblinear::{Feature, Parameter, Problem, SolverType, train};
    /// # let parameter = Parameter { solver_type: SolverType::L2rL2lossSvc, eps: 0.1, c: 1.0, class_weights: Vec::new(), p: 0.1, nu: 0.5, init_sol: None, regularize_bias: true, w_recalc: false };
    /// # let problem = Problem { solvers: vec![SolverType::L2rL2lossSvc], parameter: parameter.clone(), nr_feature: 1, bias: -1.0, labels: vec![1.0], rows: vec![vec![Feature { index: 1, value: 1.0 }]] };
    /// # let model = train(&problem, &parameter)?;
    /// assert!(model.rho.is_none());
    /// # Ok::<(), liblinear::Error>(())
    /// ```
    pub rho: Option<f64>,
    /// None for regression and one-class models.
    ///
    /// ```
    /// # use liblinear::{Feature, Parameter, Problem, SolverType, train};
    /// # let parameter = Parameter { solver_type: SolverType::L2rL2lossSvc, eps: 0.1, c: 1.0, class_weights: Vec::new(), p: 0.1, nu: 0.5, init_sol: None, regularize_bias: true, w_recalc: false };
    /// # let problem = Problem { solvers: vec![SolverType::L2rL2lossSvc], parameter: parameter.clone(), nr_feature: 1, bias: -1.0, labels: vec![1.0], rows: vec![vec![Feature { index: 1, value: 1.0 }]] };
    /// # let model = train(&problem, &parameter)?;
    /// assert_eq!(model.labels.as_deref(), Some(&[1][..]));
    /// # Ok::<(), liblinear::Error>(())
    /// ```
    pub labels: Option<Vec<i32>>,
    /// Weights in feature order, then class order.
    ///
    /// ```
    /// # use liblinear::{Feature, Parameter, Problem, SolverType, train};
    /// # let parameter = Parameter { solver_type: SolverType::L2rL2lossSvc, eps: 0.1, c: 1.0, class_weights: Vec::new(), p: 0.1, nu: 0.5, init_sol: None, regularize_bias: true, w_recalc: false };
    /// # let problem = Problem { solvers: vec![SolverType::L2rL2lossSvc], parameter: parameter.clone(), nr_feature: 1, bias: -1.0, labels: vec![1.0], rows: vec![vec![Feature { index: 1, value: 1.0 }]] };
    /// # let model = train(&problem, &parameter)?;
    /// assert_eq!(model.weights.len(), 1);
    /// # Ok::<(), liblinear::Error>(())
    /// ```
    pub weights: Vec<f64>,
    /// One score row for each input row.
    ///
    /// ```
    /// # use liblinear::{Feature, Parameter, Problem, SolverType, train};
    /// # let parameter = Parameter { solver_type: SolverType::L2rL2lossSvc, eps: 0.1, c: 1.0, class_weights: Vec::new(), p: 0.1, nu: 0.5, init_sol: None, regularize_bias: true, w_recalc: false };
    /// # let problem = Problem { solvers: vec![SolverType::L2rL2lossSvc], parameter: parameter.clone(), nr_feature: 1, bias: -1.0, labels: vec![1.0], rows: vec![vec![Feature { index: 1, value: 1.0 }]] };
    /// # let model = train(&problem, &parameter)?;
    /// assert_eq!(model.decision_values.len(), 1);
    /// # Ok::<(), liblinear::Error>(())
    /// ```
    pub decision_values: Vec<Vec<f64>>,
    /// One label or target for each input row.
    ///
    /// ```
    /// # use liblinear::{Feature, Parameter, Problem, SolverType, train};
    /// # let parameter = Parameter { solver_type: SolverType::L2rL2lossSvc, eps: 0.1, c: 1.0, class_weights: Vec::new(), p: 0.1, nu: 0.5, init_sol: None, regularize_bias: true, w_recalc: false };
    /// # let problem = Problem { solvers: vec![SolverType::L2rL2lossSvc], parameter: parameter.clone(), nr_feature: 1, bias: -1.0, labels: vec![1.0], rows: vec![vec![Feature { index: 1, value: 1.0 }]] };
    /// # let model = train(&problem, &parameter)?;
    /// assert_eq!(model.predictions.len(), 1);
    /// # Ok::<(), liblinear::Error>(())
    /// ```
    pub predictions: Vec<f64>,
}

impl Model {
    /// Return one score row for each input row.
    ///
    /// Scores use feature order, then class order.
    /// The caller supplies the bias feature when the model uses bias.
    /// The scorer leaves out a feature index above the model size.
    ///
    /// # Errors
    ///
    /// Returns an error when a one-class model has no rho.
    /// Returns an error when a feature index is zero.
    /// Returns an error when the weight vector does not hold that feature.
    /// Returns an error when the feature count overflows.
    ///
    /// ```
    /// use liblinear::{Feature, Parameter, Problem, SolverType, train};
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
    /// let problem = Problem {
    ///     solvers: vec![SolverType::L2rL2lossSvc],
    ///     parameter: parameter.clone(),
    ///     nr_feature: 1,
    ///     bias: -1.0,
    ///     labels: vec![1.0],
    ///     rows: vec![vec![Feature { index: 1, value: 1.0 }]],
    /// };
    /// let model = train(&problem, &parameter)?;
    /// assert_eq!(model.decision_values(&problem)?.len(), 1);
    /// # Ok::<(), liblinear::Error>(())
    /// ```
    pub fn decision_values(&self, problem: &Problem) -> Result<Vec<Vec<f64>>, Error> {
        let n = self
            .nr_feature
            .checked_add(usize::from(self.bias >= 0.0))
            .ok_or_else(Error::overflow)?;
        let columns = if self.nr_class == 2 && self.parameter.solver_type != SolverType::McsvmCs {
            1
        } else {
            self.nr_class
        };
        let rho = if self.parameter.solver_type == SolverType::OneclassSvm {
            Some(self.rho.ok_or(Error::MissingRho)?)
        } else {
            None
        };
        // One length check covers every feature row below.
        let width = n.checked_mul(columns).ok_or_else(Error::overflow)?;
        let weights = self
            .weights
            .get(..width)
            .ok_or_else(|| Error::range(width, self.weights.len()))?;
        let mut result = Vec::with_capacity(problem.rows.len());
        for row in &problem.rows {
            let mut values = vec![0.0; columns];
            if columns != 0 {
                for feature in row {
                    if feature.index == 0 {
                        return Err(Error::input(
                            0,
                            "Feature index 0 is not valid. Start feature indices at 1.",
                        ));
                    }
                    if feature.index <= n {
                        let start = feature.index.wrapping_sub(1).wrapping_mul(columns);
                        let end = start.wrapping_add(columns);
                        #[expect(
                            clippy::indexing_slicing,
                            reason = "decision_values checks that weights holds n times columns entries and the feature index is in 1..=n"
                        )]
                        let row_weights = &weights[start..end];
                        for (value, &weight) in values.iter_mut().zip(row_weights) {
                            *value += weight * feature.value;
                        }
                    }
                }
            }
            if let Some(rho) = rho {
                let length = values.len();
                let score = values.first_mut().ok_or_else(|| Error::index(0, length))?;
                *score -= rho;
            }
            result.push(values);
        }
        Ok(result)
    }

    /// Return one label or target per row from a model made by `train`.
    ///
    /// A tie selects the first class.
    /// A binary score of zero selects the second class.
    ///
    /// # Errors
    ///
    /// Returns an error when a one-class model has no rho.
    /// Returns an error when a classification model has no labels.
    /// Returns an error when a score or a label index is outside its array.
    ///
    /// ```
    /// use liblinear::{Feature, Parameter, Problem, SolverType, train};
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
    /// let problem = Problem {
    ///     solvers: vec![SolverType::L2rL2lossSvc],
    ///     parameter: parameter.clone(),
    ///     nr_feature: 1,
    ///     bias: -1.0,
    ///     labels: vec![1.0],
    ///     rows: vec![vec![Feature { index: 1, value: 1.0 }]],
    /// };
    /// let model = train(&problem, &parameter)?;
    /// assert_eq!(model.predict(&problem)?.len(), 1);
    /// # Ok::<(), liblinear::Error>(())
    /// ```
    #[expect(
        clippy::bool_to_int_with_if,
        reason = "The C branch also selects class 1 for NaN"
    )]
    pub fn predict(&self, problem: &Problem) -> Result<Vec<f64>, Error> {
        if matches!(
            self.parameter.solver_type,
            SolverType::L2rL2lossSvr | SolverType::L2rL2lossSvrDual | SolverType::L2rL1lossSvrDual
        ) {
            return column_zero(&self.decision_values(problem)?);
        }
        if self.parameter.solver_type == SolverType::OneclassSvm {
            let scores = self.decision_values(problem)?;
            let mut predictions = Vec::with_capacity(scores.len());
            for values in &scores {
                let score = first_score(values)?;
                predictions.push(if score > 0.0 { 1.0 } else { -1.0 });
            }
            return Ok(predictions);
        }
        let labels = self.labels.as_ref().ok_or(Error::MissingLabels)?;
        let scores = self.decision_values(problem)?;
        let mut predictions = Vec::with_capacity(scores.len());
        for values in &scores {
            let class = if self.nr_class == 2 {
                let score = first_score(values)?;
                if score > 0.0 { 0 } else { 1 }
            } else {
                let mut best = 0usize;
                let mut best_score = first_score(values)?;
                for class in 1..self.nr_class {
                    let score = values
                        .get(class)
                        .copied()
                        .ok_or_else(|| Error::index(class, values.len()))?;
                    if score > best_score {
                        best = class;
                        best_score = score;
                    }
                }
                best
            };
            let label = labels
                .get(class)
                .copied()
                .ok_or_else(|| Error::index(class, labels.len()))?;
            predictions.push(f64::from(label));
        }
        Ok(predictions)
    }
}

fn first_score(values: &[f64]) -> Result<f64, Error> {
    values
        .first()
        .copied()
        .ok_or_else(|| Error::index(0, values.len()))
}

fn column_zero(scores: &[Vec<f64>]) -> Result<Vec<f64>, Error> {
    let mut predictions = Vec::with_capacity(scores.len());
    for values in scores {
        predictions.push(first_score(values)?);
    }
    Ok(predictions)
}

#[cfg(kani)]
mod kani_proofs {
    use crate::{Feature, Parameter, Problem, SolverType};

    fn base_parameter(solver: SolverType) -> Parameter {
        Parameter {
            solver_type: solver,
            eps: 0.1,
            c: 1.0,
            class_weights: Vec::new(),
            p: 0.1,
            nu: 0.5,
            init_sol: None,
            regularize_bias: true,
            w_recalc: false,
        }
    }

    fn bare_model(nr_feature: usize, nr_class: usize, weights: Vec<f64>) -> super::Model {
        super::Model {
            parameter: base_parameter(SolverType::McsvmCs),
            nr_class,
            nr_feature,
            bias: -1.0,
            rho: None,
            labels: None,
            weights,
            decision_values: Vec::new(),
            predictions: Vec::new(),
        }
    }

    fn bare_problem(rows: Vec<Vec<Feature>>) -> Problem {
        let n = rows.len();
        Problem {
            solvers: vec![SolverType::McsvmCs],
            parameter: base_parameter(SolverType::McsvmCs),
            nr_feature: 1,
            bias: -1.0,
            labels: vec![1.0; n],
            rows,
        }
    }

    fn zeros(width: usize) -> Vec<f64> {
        let mut weights = Vec::new();
        let mut i = 0usize;
        while i < width {
            weights.push(0.0);
            i = match i.checked_add(1) {
                Some(next) => next,
                None => break,
            };
        }
        weights
    }

    // Empty rows stop before the feature loop. The product check still runs.
    // Solver 4 uses one column per class.
    #[kani::proof]
    #[kani::unwind(4)]
    fn prove_weight_product_errors_on_overflow() {
        let nr_feature: usize = kani::any();
        let nr_class: usize = kani::any();
        let model = bare_model(nr_feature, nr_class, Vec::new());
        let got = model.decision_values(&bare_problem(Vec::new()));
        match nr_feature.checked_mul(nr_class) {
            None => assert!(got.is_err()),
            Some(0) => assert!(got.is_ok()),
            Some(_) => assert!(got.is_err()),
        }
    }

    // These are the index expressions in `decision_values`.
    // A product that fits keeps the slice inside it.
    // With a symbolic column count the solver does not finish on the CI
    // runner (40 min at 64 bits, over 20 min at 16 bits), so CI proves the
    // fixed column counts below instead. Run this one with `--cfg kani_slow`.
    #[cfg(kani_slow)]
    #[kani::proof]
    #[kani::unwind(2)]
    fn prove_weight_slice_stays_inside() {
        let n: usize = kani::any();
        let columns: usize = kani::any();
        let index: usize = kani::any();
        kani::assume(index >= 1);
        kani::assume(index <= n);
        let Some(width) = n.checked_mul(columns) else {
            return;
        };
        let start = index.wrapping_sub(1).wrapping_mul(columns);
        let end = start.wrapping_add(columns);
        assert_eq!(
            index
                .checked_sub(1)
                .and_then(|base| base.checked_mul(columns)),
            Some(start)
        );
        assert_eq!(
            index
                .checked_sub(1)
                .and_then(|base| base.checked_mul(columns))
                .and_then(|base| base.checked_add(columns)),
            Some(end)
        );
        assert!(start <= end);
        assert!(end <= width);
    }

    // The slice stays inside for any n and index at a fixed column count.
    fn check_slice(columns: usize) {
        let n: usize = kani::any();
        let index: usize = kani::any();
        kani::assume(index >= 1);
        kani::assume(index <= n);
        let Some(width) = n.checked_mul(columns) else {
            return;
        };
        let start = index.wrapping_sub(1).wrapping_mul(columns);
        let end = start.wrapping_add(columns);
        assert_eq!(
            index
                .checked_sub(1)
                .and_then(|base| base.checked_mul(columns)),
            Some(start)
        );
        assert_eq!(
            index
                .checked_sub(1)
                .and_then(|base| base.checked_mul(columns))
                .and_then(|base| base.checked_add(columns)),
            Some(end)
        );
        assert!(start <= end);
        assert!(end <= width);
    }

    #[kani::proof]
    fn prove_weight_slice_stays_inside_1_columns() {
        check_slice(1);
    }

    #[kani::proof]
    fn prove_weight_slice_stays_inside_2_columns() {
        check_slice(2);
    }

    #[kani::proof]
    fn prove_weight_slice_stays_inside_3_columns() {
        check_slice(3);
    }

    #[kani::proof]
    fn prove_weight_slice_stays_inside_4_columns() {
        check_slice(4);
    }

    #[kani::proof]
    fn prove_weight_slice_stays_inside_5_columns() {
        check_slice(5);
    }

    #[kani::proof]
    fn prove_weight_slice_stays_inside_6_columns() {
        check_slice(6);
    }

    #[kani::proof]
    fn prove_weight_slice_stays_inside_7_columns() {
        check_slice(7);
    }

    #[kani::proof]
    fn prove_weight_slice_stays_inside_8_columns() {
        check_slice(8);
    }

    // One row at a fixed shape: any valid index gives decision values.
    // A symbolic shape ran the CI runner out of resources, so each of the
    // nine shapes with 1 to 3 features and 1 to 3 columns has its own proof.
    fn check_decision_values(n: usize, columns: usize) {
        let index: u8 = kani::any();
        kani::assume(index >= 1 && usize::from(index) <= n);
        let model = bare_model(n, columns, zeros(n * columns));
        let row = vec![Feature {
            index: usize::from(index),
            value: 1.0,
        }];
        let got = model.decision_values(&bare_problem(vec![row]));
        assert!(got.is_ok());
    }

    #[kani::proof]
    #[kani::unwind(12)]
    fn prove_decision_values_accepts_an_index_1_by_1() {
        check_decision_values(1, 1);
    }

    #[kani::proof]
    #[kani::unwind(12)]
    fn prove_decision_values_accepts_an_index_1_by_2() {
        check_decision_values(1, 2);
    }

    #[kani::proof]
    #[kani::unwind(12)]
    fn prove_decision_values_accepts_an_index_1_by_3() {
        check_decision_values(1, 3);
    }

    #[kani::proof]
    #[kani::unwind(12)]
    fn prove_decision_values_accepts_an_index_2_by_1() {
        check_decision_values(2, 1);
    }

    #[kani::proof]
    #[kani::unwind(12)]
    fn prove_decision_values_accepts_an_index_2_by_2() {
        check_decision_values(2, 2);
    }

    #[kani::proof]
    #[kani::unwind(12)]
    fn prove_decision_values_accepts_an_index_2_by_3() {
        check_decision_values(2, 3);
    }

    #[kani::proof]
    #[kani::unwind(12)]
    fn prove_decision_values_accepts_an_index_3_by_1() {
        check_decision_values(3, 1);
    }

    #[kani::proof]
    #[kani::unwind(12)]
    fn prove_decision_values_accepts_an_index_3_by_2() {
        check_decision_values(3, 2);
    }

    #[kani::proof]
    #[kani::unwind(12)]
    fn prove_decision_values_accepts_an_index_3_by_3() {
        check_decision_values(3, 3);
    }
}
