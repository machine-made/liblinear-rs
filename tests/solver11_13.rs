// SPDX-License-Identifier: BSD-3-Clause
use crate::parity_common::*;
use crate::{SolverType, read_problem, train, train_with_svr_threshold};
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
            train_with_svr_threshold(&problem, &parameter, true)
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

const SOLVERS: [SolverType; 3] = [
    SolverType::L2rL2lossSvr,
    SolverType::L2rL2lossSvrDual,
    SolverType::L2rL1lossSvrDual,
];

#[test]
fn solver11_13_matches_every_frozen_case() {
    for (solver, expected_count) in SOLVERS.into_iter().zip([13, 12, 12]) {
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
fn changed_epsilon_threshold_is_rejected() {
    for solver in [SOLVERS[1], SOLVERS[2]] {
        let error =
            corpus(solver, true, false).expect_err("The changed threshold must fail parity.");
        writeln!(
            io::stdout().lock(),
            "threshold mutation rejected for solver {}: {error}",
            solver as i32
        )
        .unwrap();
    }
}

#[test]
#[ignore = "This check must fail. Run it alone to show the mutation failure."]
fn changed_epsilon_threshold_matches_reference() {
    corpus(SOLVERS[1], true, false).unwrap_or_else(|error| panic!("{error}"));
}

#[test]
fn one_row_and_zero_feature_cases() {
    let rules = load(&root().join("tolerances.json"));
    for number in [11, 12, 13] {
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
            assert_eq!(model.nr_class, 2);
            assert!(model.labels.is_none());
            assert!(model.rho.is_none());
            if kind == "one_row" {
                assert_eq!(model.predictions.len(), 1);
            }
        }
    }
}

#[test]
fn six_extra_c_problems() {
    let fixtures = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
    let extra_path = fixtures.join("svr_extra.json");
    let metadata = load(&fixtures.join("svr_extra.build.json"));
    assert_eq!(metadata["output_sha256"], sha256(&extra_path));
    let repo = fixtures.parent().unwrap().parent().unwrap();
    for (path, hash) in metadata["source_hashes"].as_object().unwrap() {
        assert_eq!(hash, &sha256(&repo.join(path)));
    }
    let reference = load(&extra_path);
    let entries = reference["problems"].as_array().unwrap();
    assert_eq!(entries.len(), 6);
    let rules = load(&root().join("tolerances.json"));
    for entry in entries {
        let name = entry["name"].as_str().unwrap();
        let path = fixtures.join(format!("svr_{name}.txt"));
        assert_eq!(entry["input_sha256"], sha256(&path));
        let problem = read_problem(&path).unwrap();
        for (i, solver) in SOLVERS.into_iter().enumerate() {
            let mut parameter = problem.parameter.clone();
            parameter.solver_type = solver;
            let model = train(&problem, &parameter).unwrap();
            compare(
                &run_json(&model),
                &entry["runs"][i],
                name,
                "runs[]",
                &rules["fields"],
                &mut 0,
            )
            .unwrap();
            let repeated = train(&problem, &parameter).unwrap();
            compare(
                &run_json(&repeated),
                &entry["runs"][i],
                name,
                "runs[]",
                &rules["fields"],
                &mut 0,
            )
            .unwrap();
        }
    }
    writeln!(
        io::stdout().lock(),
        "svr extra: 6 of 6 problems equal for solvers 11, 12 and 13"
    )
    .unwrap();
}

#[test]
fn regression_parameter_rules() {
    for number in [11, 12, 13] {
        let mut problem =
            read_problem(root().join(format!("corpus/s{number:02}_one_row.txt"))).unwrap();
        let mut parameter = problem.parameter.clone();
        parameter.p = -0.1;
        if number == 11 {
            assert_eq!(
                train(&problem, &parameter).unwrap_err().to_string(),
                "p < 0. Set p to zero or a positive number."
            );
        } else {
            // The frozen source checks negative p only for solver 11.
            assert!(train(&problem, &parameter).is_ok());
        }
        parameter.p = 0.0;
        problem.labels[0] = 2.75;
        let model = train(&problem, &parameter).unwrap();
        assert_eq!(model.predictions.len(), 1);
        parameter.class_weights = vec![(2, 1000.0)];
        let weighted = train(&problem, &parameter).unwrap();
        assert_eq!(run_json(&model), run_json(&weighted));
        parameter.w_recalc = true;
        assert!(
            train(&problem, &parameter)
                .unwrap_err()
                .to_string()
                .contains("Recalculating w")
        );
        parameter.w_recalc = false;
        parameter.init_sol = Some(vec![0.0; model.weights.len()]);
        if number == 11 {
            parameter.init_sol = Some(vec![0.25; model.weights.len()]);
            parameter.eps = 1e9;
            let initial_model = train(&problem, &parameter).unwrap();
            for value in initial_model.weights {
                assert_eq!(value.to_bits(), 0.25_f64.to_bits());
            }
            parameter.eps = problem.parameter.eps;
            parameter.init_sol = Some(vec![]);
            assert!(
                train(&problem, &parameter)
                    .unwrap_err()
                    .to_string()
                    .contains("initial weight")
            );
        } else {
            assert!(
                train(&problem, &parameter)
                    .unwrap_err()
                    .to_string()
                    .contains("Initial-solution")
            );
        }
        parameter.init_sol = None;
        parameter.regularize_bias = false;
        problem.bias = 1.0;
        for row in &mut problem.rows {
            row.push(crate::Feature {
                index: problem.nr_feature + 1,
                value: 1.0,
            });
        }
        if number == 11 {
            assert!(train(&problem, &parameter).is_ok());
        } else {
            assert!(
                train(&problem, &parameter)
                    .unwrap_err()
                    .to_string()
                    .contains("-R option supported only")
            );
        }
        parameter.regularize_bias = true;
        parameter.p = f64::NAN;
        assert!(
            train(&problem, &parameter)
                .unwrap_err()
                .to_string()
                .contains("finite bias, p and nu")
        );
        parameter.p = 0.0;
        problem.labels[0] = f64::INFINITY;
        assert!(
            train(&problem, &parameter)
                .unwrap_err()
                .to_string()
                .contains("finite number")
        );
        problem.rows.clear();
        problem.labels.clear();
        assert_eq!(
            train(&problem, &parameter).unwrap_err().to_string(),
            "Add at least one data row."
        );
    }
}
