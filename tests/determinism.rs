// SPDX-License-Identifier: BSD-3-Clause
//! Bit identity of one train and predict call for each solver family.
//! The library starts no threads. This test starts four.

use liblinear::{Feature, Parameter, Problem, SolverType, train};

type RunError = Box<dyn std::error::Error + Send + Sync>;

const INDEX: [usize; 4] = [1, 2, 3, 4];
const ROWS: [[f64; 4]; 8] = [
    [0.0, 0.25, -0.5, 0.75],
    [1.0, -0.25, 0.5, -0.75],
    [-1.0, 0.5, 0.0, 0.25],
    [0.75, -0.5, 1.0, -0.25],
    [-0.75, 0.0, 0.25, 0.5],
    [0.5, 1.0, -1.0, 0.0],
    [-0.25, 0.75, 0.5, -1.0],
    [0.25, -1.0, -0.25, 1.0],
];
const LABELS: [f64; 8] = [1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0];
const TARGETS: [f64; 8] = [0.0, 0.25, 0.5, 0.75, 1.0, -0.25, -0.5, -0.75];

fn push_f64(out: &mut Vec<u64>, values: &[f64]) {
    for value in values {
        out.push(value.to_bits());
    }
}

fn problem_for(solver: SolverType) -> (Problem, Parameter) {
    let mut rows = Vec::with_capacity(ROWS.len());
    for row in ROWS {
        let mut entries = Vec::with_capacity(INDEX.len());
        for (&index, value) in INDEX.iter().zip(row) {
            entries.push(Feature { index, value });
        }
        rows.push(entries);
    }
    let labels = if solver == SolverType::L2rL2lossSvr {
        TARGETS.to_vec()
    } else {
        LABELS.to_vec()
    };
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
        nr_feature: INDEX.len(),
        bias: -1.0,
        labels,
        rows,
    };
    (problem, parameter)
}

fn sample() -> Result<Vec<u64>, RunError> {
    let mut bits = Vec::new();
    // Families 0, 1, 2, 4, 5, 11 and 21.
    for solver in [
        SolverType::L2rLr,
        SolverType::L2rL2lossSvcDual,
        SolverType::L2rL2lossSvc,
        SolverType::McsvmCs,
        SolverType::L1rL2lossSvc,
        SolverType::L2rL2lossSvr,
        SolverType::OneclassSvm,
    ] {
        let (problem, parameter) = problem_for(solver);
        let model = train(&problem, &parameter)?;
        push_f64(&mut bits, &model.weights);
        for row in &model.decision_values {
            push_f64(&mut bits, row);
        }
        push_f64(&mut bits, &model.predictions);
        let scores = model.decision_values(&problem)?;
        for row in &scores {
            push_f64(&mut bits, row);
        }
        push_f64(&mut bits, &model.predict(&problem)?);
    }
    Ok(bits)
}

#[test]
fn one_thread_and_four_threads_match_bits() -> Result<(), RunError> {
    let serial = sample()?;
    let parallel = std::thread::scope(|scope| {
        let mut guards = Vec::new();
        for _ in 0..4 {
            guards.push(scope.spawn(sample));
        }
        let mut runs = Vec::new();
        for guard in guards {
            let run = guard.join().map_err(|_| "a worker thread panicked")?;
            runs.push(run?);
        }
        Ok::<Vec<Vec<u64>>, RunError>(runs)
    })?;
    for bits in &parallel {
        assert_eq!(bits, &serial);
    }
    Ok(())
}
