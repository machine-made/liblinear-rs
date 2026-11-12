// SPDX-License-Identifier: BSD-3-Clause
//! Check saved cases, tolerances and the two reference builds.
#![expect(
    clippy::indexing_slicing,
    reason = "Tests read known JSON fields and saved rows"
)]
#![expect(
    clippy::panic,
    reason = "A parity mismatch prints the failed comparison"
)]
#![expect(
    clippy::unwrap_used,
    reason = "Tests read known files and saved results"
)]
use std::{
    collections::BTreeSet,
    fs,
    path::{Path, PathBuf},
};

use liblinear::{Model, SolverType, read_problem};
use serde_json::Value;

fn reference() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("reference")
}

fn files(folder: &Path, extension: &str) -> Vec<PathBuf> {
    let mut paths: Vec<_> = fs::read_dir(folder)
        .unwrap()
        .map(|entry| entry.unwrap().path())
        .collect();
    paths.sort();
    assert!(
        paths
            .iter()
            .all(|path| path.extension().unwrap() == extension)
    );
    paths
}

fn load(path: &Path) -> Value {
    serde_json::from_slice(&fs::read(path).unwrap()).unwrap()
}

fn header_count(text: &str, key: &str) -> usize {
    text.lines()
        .take_while(|line| *line != "data")
        .find_map(|line| line.strip_prefix(&format!("{key} ")))
        .unwrap()
        .parse()
        .unwrap()
}

#[test]
fn parses_all_158_cases_with_header_counts() {
    let paths = files(&reference().join("corpus"), "txt");
    assert_eq!(paths.len(), 158);
    for path in paths {
        let text = fs::read_to_string(&path).unwrap();
        let problem = read_problem(&path).unwrap();
        assert_eq!(
            problem.rows.len(),
            header_count(&text, "rows"),
            "{}",
            path.display()
        );
        assert_eq!(problem.labels.len(), problem.rows.len());
        assert_eq!(
            problem.nr_feature,
            header_count(&text, "features"),
            "{}",
            path.display()
        );
        for row in &problem.rows {
            assert!(row.windows(2).all(|pair| pair[0].index < pair[1].index));
            assert!(row.iter().all(|feature| feature.index > 0
                && feature.index <= problem.nr_feature + usize::from(problem.bias >= 0.0)));
            if problem.bias >= 0.0 {
                let last = row.last().unwrap();
                assert_eq!(last.index, problem.nr_feature + 1);
                assert_eq!(last.value.to_bits(), problem.bias.to_bits());
            }
        }
    }
}

fn check_fields(value: &Value, path: &str, rules: &Value, seen: &mut BTreeSet<String>) {
    if !path.is_empty() {
        let rule = rules
            .get(path)
            .unwrap_or_else(|| panic!("Missing tolerance for {path}"));
        assert_eq!(rule["kind"], "exact", "{path}");
        assert_eq!(rule["value"], 0, "{path}");
        assert!(
            !rule["reason"].as_str().unwrap().trim().is_empty(),
            "{path}"
        );
        seen.insert(path.to_owned());
    }
    match value {
        Value::Object(fields) => {
            for (key, child) in fields {
                let child_path = if path.is_empty() {
                    key.clone()
                } else {
                    format!("{path}.{key}")
                };
                check_fields(child, &child_path, rules, seen);
            }
        }
        Value::Array(values) => {
            for child in values {
                check_fields(child, &format!("{path}[]"), rules, seen);
            }
        }
        _ => {}
    }
}

#[test]
fn tolerances_cover_every_golden_output_field() {
    let root = reference();
    let tolerances = load(&root.join("tolerances.json"));
    let spread = load(&root.join("spread.json"));
    assert_eq!(
        tolerances["change_rule"],
        "Change a tolerance only with a new measurement."
    );
    let rules = &tolerances["fields"];
    let mut seen = BTreeSet::new();
    let mut solvers = BTreeSet::new();
    let paths = files(&root.join("golden/O2"), "json");
    assert_eq!(paths.len(), 158);
    for path in paths {
        let golden = load(&path);
        check_fields(&golden, "", rules, &mut seen);
        for run in golden["runs"].as_array().unwrap() {
            let solver = run["solver"].as_i64().unwrap();
            solvers.insert(solver);
            let declared = tolerances["solver_fields"][solver.to_string()]
                .as_array()
                .unwrap();
            let actual: BTreeSet<_> = run
                .as_object()
                .unwrap()
                .keys()
                .map(String::as_str)
                .collect();
            let declared: BTreeSet<_> = declared
                .iter()
                .map(|value| value.as_str().unwrap())
                .collect();
            assert_eq!(actual, declared, "solver {solver}");
            assert_eq!(run["rho"].is_null(), solver != 21);
            assert_eq!(run["labels"].is_null(), matches!(solver, 11 | 12 | 13 | 21));
        }
    }
    assert_eq!(
        solvers,
        BTreeSet::from([0, 1, 2, 3, 4, 5, 6, 7, 11, 12, 13, 21])
    );
    assert_eq!(seen, rules.as_object().unwrap().keys().cloned().collect());
    for (field, measurement) in spread["outputs"].as_object().unwrap() {
        for metric in ["max_absolute", "max_relative"] {
            assert_eq!(
                measurement[metric].as_f64().unwrap().to_bits(),
                0.0_f64.to_bits()
            );
        }
        assert_eq!(rules[field]["measurement"], "reference/spread.json");
        assert!(rules[field]["reason"].as_str().unwrap().contains("zero"));
    }
}

fn bit_equal(left: &Value, right: &Value) {
    match (left, right) {
        (Value::Number(a), Value::Number(b)) => {
            assert_eq!(a, b);
            assert_eq!(a.as_f64().unwrap().to_bits(), b.as_f64().unwrap().to_bits());
        }
        (Value::Array(a), Value::Array(b)) => {
            assert_eq!(a.len(), b.len());
            for (a, b) in a.iter().zip(b) {
                bit_equal(a, b);
            }
        }
        (Value::Object(a), Value::Object(b)) => {
            assert_eq!(a.keys().collect::<Vec<_>>(), b.keys().collect::<Vec<_>>());
            for (key, a) in a {
                bit_equal(a, &b[key]);
            }
        }
        _ => assert_eq!(left, right),
    }
}

fn doubles(value: &Value) -> Vec<f64> {
    value
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap())
        .collect()
}

fn model(run: &Value, parameter: &liblinear::Parameter) -> Model {
    let mut parameter = parameter.clone();
    parameter.solver_type =
        SolverType::try_from(i32::try_from(run["solver"].as_i64().unwrap()).unwrap()).unwrap();
    parameter.c = run["C"].as_f64().unwrap();
    parameter.eps = run["eps"].as_f64().unwrap();
    Model {
        parameter,
        nr_class: usize::try_from(run["nr_class"].as_u64().unwrap()).unwrap(),
        nr_feature: usize::try_from(run["nr_feature"].as_u64().unwrap()).unwrap(),
        bias: run["bias"].as_f64().unwrap(),
        rho: run["rho"].as_f64(),
        labels: run["labels"].as_array().map(|labels| {
            labels
                .iter()
                .map(|v| i32::try_from(v.as_i64().unwrap()).unwrap())
                .collect()
        }),
        weights: doubles(&run["weights"]),
        decision_values: run["decision_values"]
            .as_array()
            .unwrap()
            .iter()
            .map(doubles)
            .collect(),
        predictions: doubles(&run["predictions"]),
    }
}

#[test]
fn loads_all_316_goldens_and_checks_o0_o2_equal() {
    let root = reference();
    let paths = files(&root.join("golden/O2"), "json");
    let other = files(&root.join("golden/O0"), "json");
    let cases = files(&root.join("corpus"), "txt");
    assert_eq!(paths.len() + other.len(), 316);
    assert_eq!(paths.len(), other.len());
    assert_eq!(paths.len(), cases.len());
    let mut loaded = 0;
    for ((path, other), case) in paths.iter().zip(&other).zip(&cases) {
        assert_eq!(path.file_name(), other.file_name());
        assert_eq!(path.file_stem(), case.file_stem());
        let a = load(path);
        let b = load(other);
        loaded += 2;
        bit_equal(&a, &b);
        assert_eq!(fs::read(path).unwrap(), fs::read(other).unwrap());
        let problem = read_problem(case).unwrap();
        let runs = a["runs"].as_array().unwrap();
        assert_eq!(runs.len(), problem.solvers.len());
        for (run, &solver) in runs.iter().zip(&problem.solvers) {
            let model = model(run, &problem.parameter);
            assert_eq!(model.parameter.solver_type, solver);
            assert_eq!(model.parameter.c.to_bits(), problem.parameter.c.to_bits());
            assert_eq!(
                model.parameter.eps.to_bits(),
                problem.parameter.eps.to_bits()
            );
            assert_eq!(model.bias.to_bits(), problem.bias.to_bits());
            assert_eq!(model.nr_feature, problem.nr_feature);
            let columns = if model.nr_class == 2 && solver != SolverType::McsvmCs {
                1
            } else {
                model.nr_class
            };
            assert!(columns > 0);
            assert_eq!(
                model.weights.len(),
                (model.nr_feature + usize::from(model.bias >= 0.0)) * columns
            );
            assert_eq!(model.decision_values.len(), problem.rows.len());
            assert!(model.decision_values.iter().all(|row| row.len() == columns));
            assert_eq!(model.predictions.len(), problem.rows.len());
            if let Some(labels) = model.labels {
                assert_eq!(labels.len(), model.nr_class);
            }
        }
    }
    assert_eq!(loaded, 316);
}
