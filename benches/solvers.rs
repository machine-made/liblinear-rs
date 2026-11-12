// SPDX-License-Identifier: BSD-3-Clause
//! Time train and predict on one fixed problem.
//!
//! Integer loops build every input. The clock does not seed a value.

#![expect(
    clippy::arithmetic_side_effects,
    reason = "Fixed sizes build the problem rows"
)]
#![expect(
    clippy::expect_used,
    reason = "The fixed problem is valid by construction"
)]

use std::hint::black_box;
use std::time::Duration;

use criterion::{Criterion, criterion_group, criterion_main};
use liblinear::{Feature, Parameter, Problem, SolverType, train};

const ROWS: usize = 32;
const FEATURES: usize = 8;

fn fixed_problem(solver: SolverType) -> (Problem, Parameter) {
    let mut rows = Vec::with_capacity(ROWS);
    let mut labels = Vec::with_capacity(ROWS);
    for row in 0..ROWS {
        let mut entries = Vec::with_capacity(FEATURES);
        for column in 1..=FEATURES {
            let mixed = (row * 3 + column * 5) % 17;
            let value =
                f64::from(u32::try_from(mixed).expect("fixed index fits in u32")) * 0.125 - 1.0;
            entries.push(Feature {
                index: column,
                value,
            });
        }
        rows.push(entries);
        if solver == SolverType::L2rL2lossSvr {
            labels.push(f64::from(u32::try_from(row).expect("fixed row fits in u32")) * 0.25);
        } else if row % 2 == 0 {
            labels.push(1.0);
        } else {
            labels.push(-1.0);
        }
    }
    let parameter = Parameter {
        solver_type: solver,
        eps: 0.1,
        c: 1.0,
        class_weights: Vec::new(),
        p: 0.1,
        nu: 0.5,
        init_sol: None,
        regularize_bias: true,
        w_recalc: false,
    };
    let problem = Problem {
        solvers: vec![solver],
        parameter: parameter.clone(),
        nr_feature: FEATURES,
        bias: -1.0,
        labels,
        rows,
    };
    (problem, parameter)
}

fn train_solver(c: &mut Criterion, solver: SolverType, name: &str) {
    let (problem, parameter) = fixed_problem(solver);
    c.bench_function(name, |bencher| {
        bencher.iter(|| {
            train(black_box(&problem), black_box(&parameter)).expect("fixed problem is valid")
        });
    });
}

fn predict_solver_2(c: &mut Criterion) {
    let (problem, parameter) = fixed_problem(SolverType::L2rL2lossSvc);
    let model = train(&problem, &parameter).expect("fixed problem is valid");
    c.bench_function("predict_solver_2", |bencher| {
        bencher.iter(|| {
            model
                .predict(black_box(&problem))
                .expect("fixed model has class labels")
        });
    });
}

fn control_sum(c: &mut Criterion) {
    let values: Vec<f64> = (0..ROWS)
        .map(|i| f64::from(u32::try_from(i).expect("fixed index fits in u32")) * 0.125)
        .collect();
    c.bench_function("control_sum_n32", |bencher| {
        bencher.iter(|| {
            let mut sum = 0.0;
            for value in black_box(&values) {
                sum += value;
            }
            sum
        });
    });
}

fn benches(c: &mut Criterion) {
    train_solver(c, SolverType::L2rLr, "train_solver_0");
    train_solver(c, SolverType::L2rL2lossSvcDual, "train_solver_1");
    train_solver(c, SolverType::L2rL2lossSvc, "train_solver_2");
    train_solver(c, SolverType::L1rL2lossSvc, "train_solver_5");
    train_solver(c, SolverType::L2rL2lossSvr, "train_solver_11");
    predict_solver_2(c);
    control_sum(c);
}

criterion_group! {
    name = solvers;
    config = Criterion::default()
        .sample_size(40)
        .warm_up_time(Duration::from_millis(300))
        .measurement_time(Duration::from_secs(2));
    targets = benches
}
criterion_main!(solvers);
