// SPDX-License-Identifier: BSD-3-Clause
// Hand-decoded bytes keep this crate free of a second parser crate.

use liblinear::{Feature, Model, Parameter, Problem, SolverType, train};
use std::sync::OnceLock;

pub const SOLVERS: [SolverType; 12] = [
    SolverType::L2rLr,
    SolverType::L2rL2lossSvcDual,
    SolverType::L2rL2lossSvc,
    SolverType::L2rL1lossSvcDual,
    SolverType::McsvmCs,
    SolverType::L1rL2lossSvc,
    SolverType::L1rLr,
    SolverType::L2rLrDual,
    SolverType::L2rL2lossSvr,
    SolverType::L2rL2lossSvrDual,
    SolverType::L2rL1lossSvrDual,
    SolverType::OneclassSvm,
];

// One byte selects a finite or nonfinite value.
const VALUES: [f64; 8] = [0.0, 1.0, -1.0, 0.5, -0.25, f64::NAN, f64::INFINITY, -0.0];

struct Cursor<'a> {
    rest: &'a [u8],
}

impl<'a> Cursor<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self { rest: data }
    }

    fn byte(&mut self) -> Option<u8> {
        let (&value, rest) = self.rest.split_first()?;
        self.rest = rest;
        Some(value)
    }

    // A count above 3 can make one input train for a long time.
    fn count(&mut self) -> Option<usize> {
        Some(usize::from(self.byte()?) % 4)
    }

    // Index 0 and indices above the row width must not panic.
    fn index(&mut self) -> Option<usize> {
        Some(usize::from(self.byte()?) % 8)
    }

    fn unit(&mut self) -> Option<f64> {
        let index = usize::from(self.byte()?) % VALUES.len();
        VALUES.get(index).copied()
    }
}

pub fn problem_from(data: &[u8]) -> Option<(Problem, Parameter)> {
    let mut cur = Cursor::new(data);
    let rows_n = cur.count()?;
    let nr_feature = cur.count()?;
    let width = cur.count()?;
    let bias = if cur.byte()? & 1 == 0 { -1.0 } else { 1.0 };
    let eps = cur.unit()?;
    let c = cur.unit()?;
    let p = cur.unit()?;
    let nu = cur.unit()?;
    let regularize_bias = cur.byte()? & 1 == 0;
    let w_recalc = cur.byte()? & 1 == 1;
    let class_weights = if cur.byte()? & 1 == 1 {
        vec![(i32::from(cur.byte()?), cur.unit()?)]
    } else {
        Vec::new()
    };
    let init_sol = if cur.byte()? & 1 == 1 {
        let n = nr_feature.checked_add(usize::from(bias >= 0.0))?;
        let take = n.min(3);
        let mut init = Vec::new();
        init.try_reserve_exact(take).ok()?;
        for _ in 0..take {
            init.push(cur.unit()?);
        }
        Some(init)
    } else {
        None
    };
    let mut labels = Vec::new();
    let mut rows = Vec::new();
    labels.try_reserve_exact(rows_n).ok()?;
    rows.try_reserve_exact(rows_n).ok()?;
    for _ in 0..rows_n {
        labels.push(cur.unit()?);
        let mut row = Vec::new();
        row.try_reserve_exact(width).ok()?;
        for _ in 0..width {
            row.push(Feature {
                index: cur.index()?,
                value: cur.unit()?,
            });
        }
        rows.push(row);
    }
    let parameter = Parameter {
        solver_type: SolverType::L2rL2lossSvc,
        eps,
        c,
        class_weights,
        p,
        nu,
        init_sol,
        regularize_bias,
        w_recalc,
    };
    let problem = Problem {
        solvers: vec![parameter.solver_type],
        parameter: parameter.clone(),
        nr_feature,
        bias,
        labels,
        rows,
    };
    Some((problem, parameter))
}

pub fn rows_from(data: &[u8]) -> Option<Vec<Vec<Feature>>> {
    let mut cur = Cursor::new(data);
    let rows_n = cur.count()?;
    let mut rows = Vec::new();
    rows.try_reserve_exact(rows_n).ok()?;
    for _ in 0..rows_n {
        let width = cur.count()?;
        let mut row = Vec::new();
        row.try_reserve_exact(width).ok()?;
        for _ in 0..width {
            row.push(Feature {
                index: cur.index()?,
                value: cur.unit()?,
            });
        }
        rows.push(row);
    }
    Some(rows)
}

pub fn fixed_models() -> &'static [Model] {
    static MODELS: OnceLock<Vec<Model>> = OnceLock::new();
    MODELS.get_or_init(build_fixed_models).as_slice()
}

// Classification, regression and one-class each take one score path.
fn build_fixed_models() -> Vec<Model> {
    let specs = [
        (SolverType::L2rL2lossSvc, 1.0_f64),
        (SolverType::L2rL2lossSvr, 0.5),
        (SolverType::OneclassSvm, 1.0),
    ];
    let mut models = Vec::new();
    for (solver, label) in specs {
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
            nr_feature: 1,
            bias: -1.0,
            labels: vec![label],
            rows: vec![vec![Feature {
                index: 1,
                value: 1.0,
            }]],
        };
        if let Ok(model) = train(&problem, &parameter) {
            models.push(model);
        }
    }
    models
}

pub fn predict_rows(rows: Vec<Vec<Feature>>) {
    let parameter = Parameter {
        solver_type: SolverType::L2rL2lossSvc,
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
        solvers: vec![SolverType::L2rL2lossSvc],
        parameter,
        nr_feature: 1,
        bias: -1.0,
        labels: vec![1.0; rows.len()],
        rows,
    };
    for model in fixed_models() {
        let _ = model.predict(&problem);
    }
}
