// SPDX-License-Identifier: BSD-3-Clause
use crate::parity_common::*;
use crate::{SolverType, read_problem, train, train_with_log};
use serde_json::{Value, json};
use std::{
    collections::BTreeMap,
    fs,
    io::{self, Write},
};

fn corpus(solver: SolverType, mutated: bool, write_report: bool) -> Result<usize, String> {
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
        if !problem.solvers.contains(&solver) {
            continue;
        }
        let name = path.file_stem().unwrap().to_str().unwrap();
        let mut parameter = problem.parameter.clone();
        parameter.solver_type = solver;
        let model = if mutated {
            train_with_log(&problem, &parameter, |x| x.log2() * std::f64::consts::LN_2)
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
            .find(|run| run["solver"] == solver as i32)
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
        let report = json!({"solver": solver as i32, "status": "pass", "case_count": cases.len(), "worst_ulp": 0,
            "tolerances": {"path": "reference/tolerances.json", "sha256": sha256(&config)}, "cases": cases, "outputs": outputs});
        let folder = root.join("parity");
        fs::create_dir_all(&folder).unwrap();
        fs::write(
            folder.join(format!("solver{}.json", solver as i32)),
            format!("{}\n", serde_json::to_string_pretty(&report).unwrap()),
        )
        .unwrap();
    }
    Ok(cases.len())
}

#[test]
fn solver07_matches_every_frozen_case() {
    for (solver, expected_count) in [(SolverType::L2rLr, 14), (SolverType::L2rLrDual, 13)] {
        let n = corpus(solver, false, true).unwrap_or_else(|error| panic!("{error}"));
        assert_eq!(n, expected_count);
        writeln!(
            io::stdout().lock(),
            "solver {}: {n} of {n} cases equal, worst 0 ULP",
            solver as i32
        )
        .unwrap();
    }
}

#[test]
fn different_logarithm_is_rejected() {
    for solver in [SolverType::L2rLr, SolverType::L2rLrDual] {
        let error =
            corpus(solver, true, false).expect_err("The changed logarithm must fail parity.");
        writeln!(
            io::stdout().lock(),
            "logarithm mutation rejected for solver {}: {error}",
            solver as i32
        )
        .unwrap();
    }
}

#[test]
#[ignore = "This check must fail. Run it alone to show the mutation failure."]
fn different_logarithm_matches_reference() {
    corpus(SolverType::L2rLr, true, false).unwrap_or_else(|error| panic!("{error}"));
}

#[test]
fn one_row_and_zero_feature_cases() {
    let rules = load(&root().join("tolerances.json"));
    for number in [0, 7] {
        for kind in ["one_row", "zero_feature"] {
            let name = format!("s{number:02}_{kind}");
            let problem = read_problem(root().join(format!("corpus/{name}.txt"))).unwrap();
            let model = train(&problem, &problem.parameter).unwrap();
            let golden = load(&root().join(format!("golden/O2/{name}.json")));
            compare(
                &run_json(&model),
                &golden["runs"][0],
                &name,
                "runs[]",
                &rules["fields"],
                &mut 0,
            )
            .unwrap();
            if kind == "one_row" {
                assert_eq!(model.nr_class, 1);
                assert_eq!(model.predictions.len(), 1);
                assert_eq!(model.predictions[0].to_bits(), problem.labels[0].to_bits());
            }
        }
    }
}

#[test]
fn extra_c_cases_and_initial_weights() {
    let fixtures = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
    let rules = load(&root().join("tolerances.json"));
    for (name, initial) in [
        ("lr_multiclass", false),
        ("lr_fallback", false),
        ("lr_binary_labels", false),
        ("lr_free_bias", false),
        ("lr_free_bias", true),
    ] {
        let problem = read_problem(fixtures.join(format!("{name}.txt"))).unwrap();
        let suffix = if initial { "_initial" } else { "" };
        let reference = load(&fixtures.join(format!("{name}{suffix}.json")));
        for (i, &solver) in problem.solvers.iter().enumerate() {
            let mut parameter = problem.parameter.clone();
            parameter.solver_type = solver;
            if initial {
                parameter.init_sol = Some((0..12).map(|j| f64::from(j) / 100.0).collect());
            }
            let model = train(&problem, &parameter).unwrap();
            compare(
                &run_json(&model),
                &reference["runs"][i],
                name,
                "runs[]",
                &rules["fields"],
                &mut 0,
            )
            .unwrap();
            let repeated = train(&problem, &parameter).unwrap();
            compare(
                &run_json(&repeated),
                &reference["runs"][i],
                name,
                "runs[]",
                &rules["fields"],
                &mut 0,
            )
            .unwrap();
            assert_eq!(
                model.labels,
                Some(if name == "lr_binary_labels" {
                    vec![1, -1]
                } else {
                    vec![7, -4, 3]
                })
            );
        }
    }
}

#[test]
fn logistic_parameter_rules() {
    let problem = read_problem(root().join("corpus/s00_bias_on.txt")).unwrap();
    for solver in [SolverType::L2rLr, SolverType::L2rLrDual] {
        let mut parameter = problem.parameter.clone();
        parameter.solver_type = solver;
        parameter.w_recalc = true;
        assert!(
            train(&problem, &parameter)
                .unwrap_err()
                .to_string()
                .contains("Recalculating w")
        );
    }
    let mut parameter = problem.parameter.clone();
    parameter.solver_type = SolverType::L2rLrDual;
    parameter.regularize_bias = false;
    assert!(
        train(&problem, &parameter)
            .unwrap_err()
            .to_string()
            .contains("-R option supported only")
    );
    parameter.regularize_bias = true;
    parameter.init_sol = Some(vec![0.0; problem.nr_feature + 1]);
    assert!(
        train(&problem, &parameter)
            .unwrap_err()
            .to_string()
            .contains("Initial-solution specification supported only")
    );
}
