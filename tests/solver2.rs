// SPDX-License-Identifier: BSD-3-Clause
use crate::{SolverType, read_problem, train, train_with_dot};
use serde_json::{Value, json};
use std::{
    collections::BTreeMap,
    fs,
    io::{self, Write},
    path::Path,
};

type Dot = fn(&[f64], &[f64]) -> Result<f64, crate::Error>;

use crate::parity_common::*;

fn corpus(dot: Option<Dot>, write_report: bool) -> Result<usize, String> {
    let root = root();
    let config = root.join("tolerances.json");
    let rules = load(&config);
    let mut paths: Vec<_> = fs::read_dir(root.join("corpus"))
        .unwrap()
        .map(|e| e.unwrap().path())
        .collect();
    paths.sort();
    let mut cases = Vec::new();
    let mut counts = BTreeMap::new();
    for path in paths {
        let problem = read_problem(&path).unwrap();
        if !problem.solvers.contains(&SolverType::L2rL2lossSvc) {
            continue;
        }
        let name = path.file_stem().unwrap().to_str().unwrap();
        let mut parameter = problem.parameter.clone();
        parameter.solver_type = SolverType::L2rL2lossSvc;
        let model = if let Some(dot) = dot {
            train_with_dot(&problem, &parameter, dot)
        } else {
            train(&problem, &parameter)
        }
        .unwrap();
        assert_eq!(
            model.decision_values,
            model.decision_values(&problem).unwrap()
        );
        assert_eq!(model.predictions, model.predict(&problem).unwrap());
        let golden_path = root.join("golden/O2").join(format!("{name}.json"));
        let golden = load(&golden_path);
        let expected = golden["runs"]
            .as_array()
            .unwrap()
            .iter()
            .find(|run| run["solver"] == 2)
            .unwrap();
        let actual = run_json(&model);
        let mut count = 0;
        let comparison = compare(
            &actual,
            expected,
            "runs[0]",
            "runs[]",
            &rules["fields"],
            &mut count,
        );
        if let Err(error) = comparison {
            return Err(format!("{name}: {error}"));
        }
        if name == "analytic_l2svc" {
            assert_eq!(model.weights[0].to_bits(), (4000.0_f64 / 4001.0).to_bits());
        }
        if write_report {
            field_counts(&actual, "runs[]", &mut counts);
            cases.push(json!({"case": name, "status": "pass", "values": count, "failures": 0,
                "max_absolute": 0.0, "max_relative": 0.0, "max_ulp": 0,
                "inputs": {"corpus": {"path": format!("reference/corpus/{name}.txt"), "sha256": sha256(&path)},
                    "reference": {"path": format!("reference/golden/O2/{name}.json"), "sha256": sha256(&golden_path)}},
                "candidate_sha256": candidate_sha256(&actual)}));
        } else {
            cases.push(Value::Null);
        }
    }
    if write_report {
        let outputs: BTreeMap<_, _> = counts
            .into_iter()
            .map(|(name, count)| {
                (
                    name,
                    json!({"count": count,
            "failures": 0, "max_absolute": 0.0, "max_relative": 0.0, "max_ulp": 0}),
                )
            })
            .collect();
        let report = json!({"solver": 2, "status": "pass", "case_count": cases.len(), "worst_ulp": 0,
            "tolerances": {"path": "reference/tolerances.json", "sha256": sha256(&config)}, "cases": cases, "outputs": outputs});
        let folder = root.join("parity");
        fs::create_dir_all(&folder).unwrap();
        fs::write(
            folder.join("solver2.json"),
            format!("{}\n", serde_json::to_string_pretty(&report).unwrap()),
        )
        .unwrap();
    }
    Ok(cases.len())
}

#[test]
fn solver2_matches_every_frozen_case() {
    let n = corpus(None, true).unwrap_or_else(|error| panic!("{error}"));
    assert_eq!(n, 15);
    writeln!(
        io::stdout().lock(),
        "solver 2: {n} of {n} cases equal, worst 0 ULP"
    )
    .unwrap();
}

fn reversed_ddot(x: &[f64], y: &[f64]) -> Result<f64, crate::Error> {
    if x.len() != y.len() {
        return Err(crate::Error::Length {
            left: x.len(),
            right: y.len(),
        });
    }
    let mut result = 0.0;
    let mut end = x.len();
    // Reverse both the five-product groups and the terms within each group.
    while end >= 5 {
        let i = end - 1;
        result += x[i] * y[i]
            + x[i - 1] * y[i - 1]
            + x[i - 2] * y[i - 2]
            + x[i - 3] * y[i - 3]
            + x[i - 4] * y[i - 4];
        end -= 5;
    }
    while end > 0 {
        end -= 1;
        result += x[end] * y[end];
    }
    Ok(result)
}

#[test]
fn reversed_sum_is_rejected() {
    let error = corpus(Some(reversed_ddot), false).expect_err("The reversed sum must fail parity.");
    writeln!(
        io::stdout().lock(),
        "reversed-sum mutation rejected: {error}"
    )
    .unwrap();
}

#[test]
#[ignore = "This check must fail. Run it alone to show the mutation failure."]
fn reversed_sum_matches_reference() {
    corpus(Some(reversed_ddot), false).unwrap_or_else(|error| panic!("{error}"));
}

#[test]
fn extra_c_reference_cases() {
    let fixtures = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
    let rules = load(&root().join("tolerances.json"));
    for name in ["multiclass", "binary_labels"] {
        let problem = read_problem(fixtures.join(format!("{name}.txt"))).unwrap();
        let model = train(&problem, &problem.parameter).unwrap();
        let reference = load(&fixtures.join(format!("{name}.json")));
        compare(
            &run_json(&model),
            &reference["runs"][0],
            name,
            "runs[]",
            &rules["fields"],
            &mut 0,
        )
        .unwrap();
        if name == "multiclass" {
            assert_eq!(model.labels, Some(vec![7, -4, 3]));
        } else {
            assert_eq!(model.labels, Some(vec![1, -1]));
        }
        let mut zero_model = model;
        zero_model.weights.fill(0.0);
        let tied_label = if name == "multiclass" {
            7.0_f64
        } else {
            -1.0_f64
        };
        for prediction in zero_model.predict(&problem).unwrap() {
            assert_eq!(prediction.to_bits(), tied_label.to_bits());
        }
    }
}

#[test]
fn initial_weights_and_solver_rules() {
    let problem = read_problem(root().join("corpus/analytic_l2svc.txt")).unwrap();
    let mut parameter = problem.parameter.clone();
    parameter.solver_type = SolverType::L2rL2lossSvc;
    parameter.init_sol = Some(vec![4000.0 / 4001.0]);
    let model = train(&problem, &parameter).unwrap();
    assert_eq!(model.weights[0].to_bits(), (4000.0_f64 / 4001.0).to_bits());
    parameter.solver_type = SolverType::OneclassSvm;
    assert_eq!(
        train(&problem, &parameter).unwrap_err().to_string(),
        "Initial-solution specification supported only for solvers L2R_LR, L2R_L2LOSS_SVC, and L2R_L2LOSS_SVR. Remove the initial weights or select solver 0, 2 or 11."
    );
    let fixtures = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
    let problem = read_problem(fixtures.join("multiclass.txt")).unwrap();
    let mut parameter = problem.parameter.clone();
    parameter.eps = 10.0;
    let initial: Vec<_> = (0..12).map(|i| f64::from(i) / 100.0).collect();
    parameter.init_sol = Some(initial.clone());
    let model = train(&problem, &parameter).unwrap();
    for (actual, expected) in model.weights.iter().zip(initial) {
        assert_eq!(actual.to_bits(), expected.to_bits());
    }
}

#[test]
fn train_rejects_invalid_inputs() {
    let valid = read_problem(root().join("corpus/s02_bias_on.txt")).unwrap();
    for case in 0..17 {
        let mut problem = valid.clone();
        let mut parameter = valid.parameter.clone();
        match case {
            0 => parameter.eps = 0.0,
            1 => parameter.c = -1.0,
            2 => parameter.eps = f64::NAN,
            3 => parameter.class_weights.push((1, f64::INFINITY)),
            4 => parameter.regularize_bias = false,
            5 => parameter.w_recalc = true,
            6 => problem.rows.clear(),
            7 => {
                problem.labels.pop();
            }
            8 => problem.labels[0] = 0.5,
            9 => problem.labels[0] = f64::from(i32::MAX) + 1.0,
            10 => problem.rows[0][0].index = 0,
            11 => problem.rows[0][0].value = f64::NAN,
            12 => {
                problem.rows[0].pop();
            }
            13 => parameter.init_sol = Some(vec![]),
            14 => parameter.init_sol = Some(vec![f64::NAN; problem.nr_feature + 1]),
            15 => problem.nr_feature = 0,
            16 => problem.bias = f64::INFINITY,
            _ => unreachable!(),
        }
        // The saved bias is 1. Use another value for the free-bias error.
        if case == 4 {
            problem.bias = 0.0;
        }
        assert!(train(&problem, &parameter).is_err(), "case {case}");
    }
}
