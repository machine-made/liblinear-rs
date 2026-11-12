// SPDX-License-Identifier: BSD-3-Clause
//! Class grouping and training for all 12 solvers from `linear.cpp`.
//!
//! ```
//! use liblinear::{Feature, Parameter, Problem, SolverType, train};
//! let parameter = Parameter {
//!     solver_type: SolverType::L2rLr,
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
//!     solvers: vec![SolverType::L2rLr],
//!     parameter: parameter.clone(),
//!     nr_feature: 1,
//!     bias: -1.0,
//!     labels: vec![1.0, -1.0],
//!     rows: vec![
//!         vec![Feature { index: 1, value: 1.0 }],
//!         vec![Feature { index: 1, value: -1.0 }],
//!     ],
//! };
//! let model = train(&problem, &parameter)?;
//! assert_eq!(model.nr_class, 2);
//! # Ok::<(), liblinear::Error>(())
//! ```

use crate::{
    Error, Model, Parameter, Problem, SolverType,
    blas::ddot,
    dual_l1l2_svc, dual_lr, dual_svr, l1r_l2_svc, l1r_lr,
    l2r_l2_svc::Objective,
    l2r_l2_svr, l2r_lr, mcsvm_cs, oneclass_svm,
    rand::GlibcRand,
    tron::{Dot, newton},
};

/// Train all 12 solvers and return an owned model.
///
/// Each call starts seed 1.
///
/// # Errors
///
/// Returns an error for invalid inputs.
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
/// assert_eq!(model.predictions.len(), 1);
/// # Ok::<(), liblinear::Error>(())
/// ```
pub fn train(problem: &Problem, parameter: &Parameter) -> Result<Model, Error> {
    train_with_dot(problem, parameter, ddot)
}

/// Train with a supplied dot product. This call supports the parity mutation check.
/// Use `train` for the upstream BLAS sum order.
///
/// # Errors
/// Returns the same input errors as `train`.
pub(crate) fn train_with_dot(
    problem: &Problem,
    parameter: &Parameter,
    dot: Dot,
) -> Result<Model, Error> {
    let mut stream = GlibcRand::new(1)?;
    train_core(
        problem,
        parameter,
        dot,
        &mut || stream.next_u31(),
        f64::ln,
        false,
        mcsvm_cs::sort,
        false,
    )
}

/// Train with a supplied random stream. This call supports the parity mutation check.
/// Use `train` for the reference stream with seed 1.
///
/// # Errors
/// Returns the same input errors as `train`.
#[cfg_attr(not(test), expect(dead_code, reason = "Parity tests call this hook"))]
pub(crate) fn train_with_rand(
    problem: &Problem,
    parameter: &Parameter,
    random: &mut dyn FnMut() -> u32,
) -> Result<Model, Error> {
    train_core(
        problem,
        parameter,
        ddot,
        random,
        f64::ln,
        false,
        mcsvm_cs::sort,
        false,
    )
}

/// Train with a supplied logarithm for the parity mutation check.
/// Use `train` for the standard library logarithm.
///
/// # Errors
/// Returns the same input errors as `train`.
#[cfg_attr(not(test), expect(dead_code, reason = "Parity tests call this hook"))]
pub(crate) fn train_with_log(
    problem: &Problem,
    parameter: &Parameter,
    logarithm: fn(f64) -> f64,
) -> Result<Model, Error> {
    let mut stream = GlibcRand::new(1)?;
    train_core(
        problem,
        parameter,
        ddot,
        &mut || stream.next_u31(),
        logarithm,
        false,
        mcsvm_cs::sort,
        false,
    )
}

/// Train with a changed shrinking comparison for the parity mutation check.
/// Set `mutated` to false for the source comparison.
///
/// # Errors
/// Returns the same input errors as `train`.
#[cfg_attr(not(test), expect(dead_code, reason = "Parity tests call this hook"))]
pub(crate) fn train_with_shrinking(
    problem: &Problem,
    parameter: &Parameter,
    mutated: bool,
) -> Result<Model, Error> {
    let mut stream = GlibcRand::new(1)?;
    train_core(
        problem,
        parameter,
        ddot,
        &mut || stream.next_u31(),
        f64::ln,
        mutated,
        mcsvm_cs::sort,
        false,
    )
}

/// Train with a supplied solver 4 sort for the parity check.
/// Use `train` for the source comparison.
///
/// # Errors
/// Returns the same input errors as `train`.
#[cfg_attr(not(test), expect(dead_code, reason = "Parity tests call this hook"))]
pub(crate) fn train_with_mcsvm_sort(
    problem: &Problem,
    parameter: &Parameter,
    sorter: fn(&mut [f64]),
) -> Result<Model, Error> {
    let mut stream = GlibcRand::new(1)?;
    train_core(
        problem,
        parameter,
        ddot,
        &mut || stream.next_u31(),
        f64::ln,
        false,
        sorter,
        false,
    )
}

/// Train with a changed dual SVR threshold for the parity mutation check.
/// Set `mutated` to false for the source threshold.
///
/// # Errors
/// Returns the same input errors as `train`.
#[cfg_attr(not(test), expect(dead_code, reason = "Parity tests call this hook"))]
pub(crate) fn train_with_svr_threshold(
    problem: &Problem,
    parameter: &Parameter,
    mutated: bool,
) -> Result<Model, Error> {
    let mut stream = GlibcRand::new(1)?;
    train_core(
        problem,
        parameter,
        ddot,
        &mut || stream.next_u31(),
        f64::ln,
        false,
        mcsvm_cs::sort,
        mutated,
    )
}

#[allow(clippy::too_many_lines, clippy::too_many_arguments)] // Keep the dispatch together.
fn train_core(
    problem: &Problem,
    parameter: &Parameter,
    dot: Dot,
    random: &mut dyn FnMut() -> u32,
    logarithm: fn(f64) -> f64,
    mutated: bool,
    sorter: fn(&mut [f64]),
    svr_mutated: bool,
) -> Result<Model, Error> {
    validate(problem, parameter)?;
    if parameter.solver_type == SolverType::OneclassSvm {
        return train_oneclass(problem, parameter, random, false);
    }
    if is_regression(parameter.solver_type) {
        return train_regression(problem, parameter, dot, random, svr_mutated);
    }
    let (labels, groups) = group_classes(problem)?;
    let nr_class = labels.len();
    let columns = if nr_class == 2 && parameter.solver_type != SolverType::McsvmCs {
        1
    } else {
        nr_class
    };
    let n = feature_width(problem.nr_feature, problem.bias)?;
    let size = n
        .checked_mul(columns)
        .ok_or_else(|| Error::input(0, "Reduce the model size."))?;
    if let Some(initial) = &parameter.init_sol
        && (initial.len() != size || initial.iter().any(|v| !v.is_finite()))
    {
        return Err(Error::input(
            0,
            "Set one finite initial weight for each model weight.",
        ));
    }
    let mut weighted_c = vec![parameter.c; nr_class];
    for &(label, weight) in &parameter.class_weights {
        if let Some(i) = labels.iter().position(|&value| value == label) {
            let length = weighted_c.len();
            let slot = weighted_c
                .get_mut(i)
                .ok_or_else(|| Error::index(i, length))?;
            *slot *= weight;
        }
    }
    let mut rows = Vec::with_capacity(problem.rows.len());
    let mut starts = Vec::with_capacity(nr_class);
    for group in &groups {
        starts.push(rows.len());
        for &i in group {
            let row = problem
                .rows
                .get(i)
                .ok_or_else(|| Error::index(i, problem.rows.len()))?;
            rows.push(row.as_slice());
        }
    }
    let mut weights = vec![0.0; size];
    if parameter.solver_type == SolverType::McsvmCs {
        let mut y = Vec::with_capacity(rows.len());
        for (class, group) in groups.iter().enumerate() {
            y.extend(std::iter::repeat_n(class, group.len()));
        }
        mcsvm_cs::solve(
            &rows,
            &y,
            &weighted_c,
            parameter,
            &mut weights,
            random,
            sorter,
        )?;
    } else {
        for class in 0..columns {
            let start = *starts
                .get(class)
                .ok_or_else(|| Error::index(class, starts.len()))?;
            let group_len = groups
                .get(class)
                .ok_or_else(|| Error::index(class, groups.len()))?
                .len();
            let end = start.checked_add(group_len).ok_or_else(Error::overflow)?;
            let mut y = vec![-1.0; rows.len()];
            let length = y.len();
            y.get_mut(start..end)
                .ok_or_else(|| Error::range(start, length))?
                .fill(1.0);
            let cp = *weighted_c
                .get(class)
                .ok_or_else(|| Error::index(class, weighted_c.len()))?;
            let cn = if nr_class == 2 {
                *weighted_c
                    .get(1)
                    .ok_or_else(|| Error::index(1, weighted_c.len()))?
            } else {
                parameter.c
            };
            let mut c = vec![0.0; rows.len()];
            let mut pos = 0usize;
            for (label, cost) in y.iter().zip(&mut c) {
                if *label > 0.0 {
                    pos = pos.checked_add(1).ok_or_else(Error::overflow)?;
                    *cost = cp;
                } else {
                    *cost = cn;
                }
            }
            let neg = rows.len().checked_sub(pos).ok_or_else(Error::overflow)?;
            // The input checks bound these counts to the upstream int range.
            let scale = f64::from(
                u32::try_from(pos.min(neg).max(1))
                    .map_err(|_| Error::input(0, "Reduce the row count."))?,
            );
            let count = f64::from(
                u32::try_from(rows.len()).map_err(|_| Error::input(0, "Reduce the row count."))?,
            );
            let eps = parameter.eps * scale / count;
            let mut w = vec![0.0; n];
            if let Some(initial) = &parameter.init_sol {
                for (j, slot) in w.iter_mut().enumerate() {
                    let index = weight_index(j, columns, class)?;
                    *slot = *initial
                        .get(index)
                        .ok_or_else(|| Error::index(index, initial.len()))?;
                }
            }
            match parameter.solver_type {
                SolverType::L1rL2lossSvc => {
                    l1r_l2_svc::solve(&rows, &y, &c, parameter, &mut w, eps, random, mutated)?;
                }
                SolverType::L1rLr => {
                    l1r_lr::solve(&rows, &y, &c, parameter, &mut w, eps, random, mutated)?;
                }
                SolverType::L2rLr | SolverType::L2rLrDual => {
                    let primal = parameter.solver_type == SolverType::L2rLr;
                    let iterations = if primal {
                        0
                    } else {
                        dual_lr::solve(&rows, &y, parameter, &mut w, cp, cn, random, logarithm)?
                    };
                    if primal || iterations >= 300 {
                        let tolerance = if primal { eps } else { eps * 0.1 };
                        let mut objective = l2r_lr::Objective::new(
                            &rows,
                            &y,
                            &c,
                            parameter.regularize_bias,
                            logarithm,
                        )?;
                        newton(&mut objective, &mut w, tolerance, dot)?;
                    }
                }
                _ => {
                    let primal = parameter.solver_type == SolverType::L2rL2lossSvc;
                    let iterations = if primal {
                        0
                    } else {
                        dual_l1l2_svc::solve(&rows, &y, parameter, &mut w, cp, cn, random)?
                    };
                    if primal
                        || (parameter.solver_type == SolverType::L2rL2lossSvcDual
                            && iterations >= 300)
                    {
                        let tolerance = if primal { eps } else { eps * 0.1 };
                        let mut objective =
                            Objective::new(&rows, &y, &c, parameter.regularize_bias)?;
                        newton(&mut objective, &mut w, tolerance, dot)?;
                    }
                }
            }
            for (j, &value) in w.iter().enumerate() {
                let index = weight_index(j, columns, class)?;
                let length = weights.len();
                let slot = weights
                    .get_mut(index)
                    .ok_or_else(|| Error::index(index, length))?;
                *slot = value;
            }
        }
    }
    let mut model = Model {
        parameter: parameter.clone(),
        nr_class,
        nr_feature: problem.nr_feature,
        bias: problem.bias,
        labels: Some(labels),
        rho: None,
        weights,
        decision_values: Vec::new(),
        predictions: Vec::new(),
    };
    model.decision_values = model.decision_values(problem)?;
    model.predictions = model.predict(problem)?;
    Ok(model)
}

/// Train with a changed rho update for the parity mutation check.
/// Set `mutated` to false for the source update.
///
/// # Errors
/// Returns the same input errors as `train`.
#[cfg_attr(not(test), expect(dead_code, reason = "Parity tests call this hook"))]
pub(crate) fn train_with_oneclass_rho(
    problem: &Problem,
    parameter: &Parameter,
    mutated: bool,
) -> Result<Model, Error> {
    if parameter.solver_type != SolverType::OneclassSvm {
        return train(problem, parameter);
    }
    validate(problem, parameter)?;
    let mut stream = GlibcRand::new(1)?;
    train_oneclass(problem, parameter, &mut || stream.next_u31(), mutated)
}

fn train_oneclass(
    problem: &Problem,
    parameter: &Parameter,
    random: &mut dyn FnMut() -> u32,
    mutated: bool,
) -> Result<Model, Error> {
    let mut weights = vec![0.0; problem.nr_feature];
    let rows: Vec<_> = problem.rows.iter().map(Vec::as_slice).collect();
    let rho = oneclass_svm::solve(&rows, parameter, &mut weights, random, mutated)?;
    let mut model = Model {
        parameter: parameter.clone(),
        nr_class: 2,
        nr_feature: problem.nr_feature,
        bias: problem.bias,
        labels: None,
        rho: Some(rho),
        weights,
        decision_values: Vec::new(),
        predictions: Vec::new(),
    };
    model.decision_values = model.decision_values(problem)?;
    model.predictions = model.predict(problem)?;
    Ok(model)
}

fn is_regression(solver: SolverType) -> bool {
    matches!(
        solver,
        SolverType::L2rL2lossSvr | SolverType::L2rL2lossSvrDual | SolverType::L2rL1lossSvrDual
    )
}

fn train_regression(
    problem: &Problem,
    parameter: &Parameter,
    dot: Dot,
    random: &mut dyn FnMut() -> u32,
    mutated: bool,
) -> Result<Model, Error> {
    let n = feature_width(problem.nr_feature, problem.bias)?;
    let mut weights = vec![0.0; n];
    if let Some(initial) = &parameter.init_sol {
        if initial.len() != n || initial.iter().any(|v| !v.is_finite()) {
            return Err(Error::input(
                0,
                "Set one finite initial weight for each model weight.",
            ));
        }
        weights.copy_from_slice(initial);
    }
    let rows: Vec<_> = problem.rows.iter().map(Vec::as_slice).collect();
    let primal = parameter.solver_type == SolverType::L2rL2lossSvr;
    let iterations = if primal {
        0
    } else {
        dual_svr::solve(
            &rows,
            &problem.labels,
            parameter,
            &mut weights,
            random,
            mutated,
        )?
    };
    if primal || (parameter.solver_type == SolverType::L2rL2lossSvrDual && iterations >= 300) {
        let c = vec![parameter.c; rows.len()];
        let tolerance = if primal {
            parameter.eps
        } else {
            parameter.eps * 0.001
        };
        let mut objective = l2r_l2_svr::Objective::new(
            &rows,
            &problem.labels,
            &c,
            parameter.regularize_bias,
            parameter.p,
        )?;
        newton(&mut objective, &mut weights, tolerance, dot)?;
    }
    let mut model = Model {
        parameter: parameter.clone(),
        nr_class: 2,
        nr_feature: problem.nr_feature,
        bias: problem.bias,
        labels: None,
        rho: None,
        weights,
        decision_values: Vec::new(),
        predictions: Vec::new(),
    };
    model.decision_values = model.decision_values(problem)?;
    model.predictions = model.predict(problem)?;
    Ok(model)
}

#[allow(clippy::cast_possible_truncation)] // Validation requires exact i32 labels.
fn group_classes(problem: &Problem) -> Result<(Vec<i32>, Vec<Vec<usize>>), Error> {
    let mut labels = Vec::new();
    let mut groups: Vec<Vec<usize>> = Vec::new();
    for (i, &value) in problem.labels.iter().enumerate() {
        let label = value as i32;
        if let Some(j) = labels.iter().position(|&v| v == label) {
            let length = groups.len();
            groups
                .get_mut(j)
                .ok_or_else(|| Error::index(j, length))?
                .push(i);
        } else {
            labels.push(label);
            groups.push(vec![i]);
        }
    }
    if labels == [-1, 1] {
        labels.swap(0, 1);
        groups.swap(0, 1);
    }
    Ok((labels, groups))
}

#[allow(clippy::too_many_lines)] // Keep the input checks in source order.
fn validate(problem: &Problem, parameter: &Parameter) -> Result<(), Error> {
    if !parameter.eps.is_finite() || parameter.eps <= 0.0 {
        return Err(Error::input(
            0,
            "eps <= 0 or not finite. Set eps to a finite number greater than zero.",
        ));
    }
    if !parameter.c.is_finite() || parameter.c <= 0.0 {
        return Err(Error::input(
            0,
            "C <= 0 or not finite. Set C to a finite number greater than zero.",
        ));
    }
    if parameter.solver_type == SolverType::L2rL2lossSvr && parameter.p < 0.0 {
        return Err(Error::input(
            0,
            "p < 0. Set p to zero or a positive number.",
        ));
    }
    if parameter.solver_type == SolverType::OneclassSvm && problem.bias >= 0.0 {
        return Err(Error::input(
            0,
            "prob->bias >=0, but this is ignored in ONECLASS_SVM. Set bias to a negative number for solver 21.",
        ));
    }
    if !parameter.regularize_bias && problem.bias.to_bits() != 1.0_f64.to_bits() {
        return Err(Error::input(
            0,
            "To not regularize bias, must specify -B 1 along with -R. Set bias to 1 to omit its penalty.",
        ));
    }
    if !parameter.regularize_bias
        && !matches!(
            parameter.solver_type,
            SolverType::L2rLr
                | SolverType::L2rL2lossSvc
                | SolverType::L1rL2lossSvc
                | SolverType::L1rLr
                | SolverType::L2rL2lossSvr
        )
    {
        return Err(Error::input(
            0,
            "-R option supported only for solver L2R_LR, L2R_L2LOSS_SVC, L1R_L2LOSS_SVC, L1R_LR, and L2R_L2LOSS_SVR. Apply the bias penalty or select solver 0, 2, 5, 6 or 11.",
        ));
    }
    if parameter.init_sol.is_some()
        && !matches!(
            parameter.solver_type,
            SolverType::L2rLr | SolverType::L2rL2lossSvc | SolverType::L2rL2lossSvr
        )
    {
        return Err(Error::input(
            0,
            "Initial-solution specification supported only for solvers L2R_LR, L2R_L2LOSS_SVC, and L2R_L2LOSS_SVR. Remove the initial weights or select solver 0, 2 or 11.",
        ));
    }
    if parameter.w_recalc
        && !matches!(
            parameter.solver_type,
            SolverType::L2rL2lossSvcDual | SolverType::L2rL1lossSvcDual
        )
    {
        return Err(Error::input(
            0,
            "Recalculating w in the end is only for dual solvers for L2-regularized L1/L2-loss SVM. Turn off weight recalculation or select solver 1 or 3.",
        ));
    }
    if problem.rows.is_empty() {
        return Err(Error::input(0, "Add at least one data row."));
    }
    if problem.rows.len() >= i32::MAX as usize
        || problem.nr_feature == 0
        || problem.nr_feature >= i32::MAX as usize
    {
        return Err(Error::input(
            0,
            "Use row and feature counts from 1 to 2147483646.",
        ));
    }
    if problem.labels.len() != problem.rows.len() {
        return Err(Error::input(0, "Set one label for each data row."));
    }
    if problem.labels.iter().any(|v| !v.is_finite()) {
        return Err(Error::input(
            0,
            "Set each label or target to a finite number.",
        ));
    }
    if !is_regression(parameter.solver_type)
        && parameter.solver_type != SolverType::OneclassSvm
        && problem
            .labels
            .iter()
            .any(|v| v.fract() != 0.0 || *v < f64::from(i32::MIN) || *v > f64::from(i32::MAX))
    {
        return Err(Error::input(
            0,
            "Set each class label to an integer in the i32 range.",
        ));
    }
    if !problem.bias.is_finite() || !parameter.p.is_finite() || !parameter.nu.is_finite() {
        return Err(Error::input(0, "Use finite bias, p and nu values."));
    }
    if parameter.solver_type == SolverType::OneclassSvm
        && (parameter.nu <= 0.0 || parameter.nu > 1.0)
    {
        return Err(Error::input(
            0,
            "Set nu to a number greater than zero and at most 1.",
        ));
    }
    if parameter
        .class_weights
        .iter()
        .any(|&(_, w)| !w.is_finite() || w <= 0.0)
    {
        return Err(Error::input(
            0,
            "Set each class weight to a finite number greater than zero.",
        ));
    }
    let n = feature_width(problem.nr_feature, problem.bias)?;
    for row in &problem.rows {
        let mut previous = 0;
        for feature in row {
            if feature.index <= previous || feature.index > n || !feature.value.is_finite() {
                return Err(Error::input(
                    0,
                    "Use finite features with ordered indices from 1 to the model feature count.",
                ));
            }
            previous = feature.index;
        }
        if problem.bias >= 0.0
            && !row.last().is_some_and(|feature| {
                feature.index == n && feature.value.to_bits() == problem.bias.to_bits()
            })
        {
            return Err(Error::input(
                0,
                "Append the bias feature once at the end of each row.",
            ));
        }
    }
    Ok(())
}

fn feature_width(nr_feature: usize, bias: f64) -> Result<usize, Error> {
    nr_feature
        .checked_add(usize::from(bias >= 0.0))
        .ok_or_else(Error::overflow)
}

fn weight_index(feature: usize, columns: usize, class: usize) -> Result<usize, Error> {
    feature
        .checked_mul(columns)
        .and_then(|base| base.checked_add(class))
        .ok_or_else(Error::overflow)
}

#[cfg(kani)]
mod kani_proofs {
    use crate::{Feature, Parameter, Problem, SolverType};

    fn base_parameter() -> Parameter {
        Parameter {
            solver_type: SolverType::L2rL2lossSvc,
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

    // Bias is off, so n equals the feature count.
    // An index must increase and stay in 1 through n.
    #[kani::proof]
    #[kani::unwind(8)]
    fn prove_feature_indices_increase_inside_width() {
        let n_byte: u8 = kani::any();
        let count: u8 = kani::any();
        let raw: [u8; 3] = kani::any();
        kani::assume(n_byte >= 1 && n_byte <= 3);
        kani::assume(count <= 3);
        let width = super::feature_width(usize::from(n_byte), -1.0);
        assert!(width.is_ok());
        let Ok(n) = width else {
            return;
        };
        let mut row = Vec::new();
        let mut i = 0u8;
        while i < count {
            let Some(&index) = raw.get(usize::from(i)) else {
                break;
            };
            row.push(Feature {
                index: usize::from(index),
                value: 1.0,
            });
            i = match i.checked_add(1) {
                Some(next) => next,
                None => break,
            };
        }
        let mut previous = 0usize;
        let mut ordered = true;
        for feature in &row {
            if feature.index <= previous || feature.index > n || !feature.value.is_finite() {
                ordered = false;
                break;
            }
            previous = feature.index;
        }
        let parameter = base_parameter();
        let problem = Problem {
            solvers: vec![SolverType::L2rL2lossSvc],
            parameter: parameter.clone(),
            nr_feature: usize::from(n_byte),
            bias: -1.0,
            labels: vec![1.0],
            rows: vec![row],
        };
        let got = super::validate(&problem, &parameter);
        assert_eq!(got.is_ok(), ordered);
    }
}
