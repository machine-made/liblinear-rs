// SPDX-License-Identifier: BSD-3-Clause
use crate::parity_common::*;
use crate::{Feature, Model, SolverType, read_problem, train, train_with_oneclass_rho};
use serde_json::{Value, json};
use std::{
    collections::BTreeMap,
    fs,
    io::{self, Write},
    path::Path,
};

fn corpus(mutated: bool, write_report: bool) -> Result<usize, String> {
    let solver = SolverType::OneclassSvm;
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
            train_with_oneclass_rho(&problem, &parameter, true)
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
fn solver21_matches_every_frozen_case() {
    let n = corpus(false, true).unwrap_or_else(|error| panic!("{error}"));
    assert_eq!(n, 11);
    writeln!(
        io::stdout().lock(),
        "solver 21: {n} of {n} cases equal, worst 0 ULP"
    )
    .unwrap();
}

#[test]
fn changed_rho_is_rejected() {
    let error = corpus(true, false).expect_err("The changed rho must fail parity.");
    assert!(error.contains("decision_values") || error.contains("rho"));
    writeln!(io::stdout().lock(), "rho mutation rejected: {error}").unwrap();
}

#[test]
#[ignore = "This check must fail. Run it alone to show the mutation failure."]
fn changed_rho_matches_reference() {
    corpus(true, false).unwrap_or_else(|error| panic!("{error}"));
}

#[test]
fn six_extra_c_problems() {
    let fixtures = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
    let extra = fixtures.join("oneclass_extra.json");
    let metadata = load(&fixtures.join("oneclass_extra.build.json"));
    assert_eq!(metadata["output_sha256"], sha256(&extra));
    let repo = fixtures.parent().unwrap().parent().unwrap();
    for (path, hash) in metadata["source_hashes"].as_object().unwrap() {
        assert_eq!(hash, &sha256(&repo.join(path)));
    }
    let reference = load(&extra);
    let problems = reference["problems"].as_array().unwrap();
    assert_eq!(problems.len(), 6);
    let rules = load(&root().join("tolerances.json"));
    for entry in problems {
        let name = entry["name"].as_str().unwrap();
        let path = fixtures.join(format!("oneclass_{name}.txt"));
        assert_eq!(entry["input_sha256"], sha256(&path));
        let problem = read_problem(&path).unwrap();
        for _ in 0..2 {
            let model = train(&problem, &problem.parameter).unwrap();
            compare(
                &run_json(&model),
                &entry["runs"][0],
                name,
                "runs[]",
                &rules["fields"],
                &mut 0,
            )
            .unwrap();
        }
    }
    assert_eq!(metadata["coverage"]["extra"]["oneclass_scaled"]["2437"], 1);
    assert_eq!(metadata["coverage"]["extra"]["oneclass_nu_half"]["2472"], 1);
    writeln!(io::stdout().lock(), "oneclass extra: 6 of 6 problems equal").unwrap();
}

#[test]
fn hand_made_model_subtracts_rho_and_uses_strict_sign() {
    let mut problem = read_problem(root().join("corpus/s21_one_row.txt")).unwrap();
    problem.rows = vec![
        vec![Feature {
            index: 1,
            value: 2.0,
        }],
        vec![Feature {
            index: 1,
            value: 1.0,
        }],
        vec![Feature {
            index: 1,
            value: 0.0,
        }],
    ];
    let model = Model {
        parameter: problem.parameter.clone(),
        nr_class: 2,
        nr_feature: 3,
        bias: -1.0,
        labels: None,
        rho: Some(2.0),
        weights: vec![2.0, 0.0, 0.0],
        decision_values: vec![],
        predictions: vec![],
    };
    let decisions = model.decision_values(&problem).unwrap();
    for (row, expected) in decisions.iter().zip([2.0_f64, 0.0, -2.0]) {
        assert_eq!(row.len(), 1);
        assert_eq!(row[0].to_bits(), expected.to_bits());
    }
    let predictions = model.predict(&problem).unwrap();
    for (actual, expected) in predictions.iter().zip([1.0_f64, -1.0, -1.0]) {
        assert_eq!(actual.to_bits(), expected.to_bits());
    }
    let mut missing_rho = model;
    missing_rho.rho = None;
    assert_eq!(
        missing_rho
            .decision_values(&problem)
            .unwrap_err()
            .to_string(),
        "The one-class model has no rho. Train solver 21 before you score rows."
    );
    missing_rho.parameter.solver_type = SolverType::L2rL2lossSvc;
    assert_eq!(
        missing_rho.predict(&problem).unwrap_err().to_string(),
        "The classification model has no labels. Train a classifier before you predict labels."
    );
}

#[test]
fn oneclass_parameter_rules_and_ignored_labels() {
    let mut problem = read_problem(root().join("corpus/s21_one_row.txt")).unwrap();
    let mut parameter = problem.parameter.clone();
    let baseline = run_json(&train(&problem, &parameter).unwrap());
    assert_eq!(baseline["predictions"].as_array().unwrap().len(), 1);
    problem.labels[0] = 1.25;
    parameter.class_weights = vec![(1, 1000.0)];
    parameter.p = -1.0;
    parameter.c = 123.0;
    let model = train(&problem, &parameter).unwrap();
    // The saved C field changes; the fit ignores C, p, weights and row labels.
    let mut changed = run_json(&model);
    changed["C"] = baseline["C"].clone();
    assert_eq!(changed, baseline);
    let source = train_with_oneclass_rho(&problem, &parameter, false).unwrap();
    assert_eq!(run_json(&source), run_json(&model));
    for nu in [-0.1, 0.0, 1.01] {
        parameter.nu = nu;
        assert_eq!(
            train(&problem, &parameter).unwrap_err().to_string(),
            "Set nu to a number greater than zero and at most 1."
        );
    }
    for nu in [f64::NAN, f64::INFINITY] {
        parameter.nu = nu;
        assert!(
            train(&problem, &parameter)
                .unwrap_err()
                .to_string()
                .contains("finite bias, p and nu")
        );
    }
    parameter.nu = 1.0;
    let endpoint = train(&problem, &parameter).unwrap();
    assert_eq!(endpoint.rho.unwrap().to_bits(), f64::INFINITY.to_bits());
    parameter.nu = 0.5;
    parameter.init_sol = Some(vec![0.0; 3]);
    assert!(
        train(&problem, &parameter)
            .unwrap_err()
            .to_string()
            .contains("Initial-solution")
    );
    parameter.init_sol = None;
    parameter.w_recalc = true;
    assert!(
        train(&problem, &parameter)
            .unwrap_err()
            .to_string()
            .contains("Recalculating w")
    );
    parameter.w_recalc = false;
    parameter.regularize_bias = false;
    assert!(
        train(&problem, &parameter)
            .unwrap_err()
            .to_string()
            .contains("must specify -B 1")
    );
    parameter.regularize_bias = true;
    problem.bias = 0.0;
    assert_eq!(
        train(&problem, &parameter).unwrap_err().to_string(),
        "prob->bias >=0, but this is ignored in ONECLASS_SVM. Set bias to a negative number for solver 21."
    );
    problem.bias = -1.0;
    problem.rows.clear();
    problem.labels.clear();
    assert_eq!(
        train(&problem, &parameter).unwrap_err().to_string(),
        "Add at least one data row."
    );
    assert_eq!(
        SolverType::try_from(99).unwrap_err().to_string(),
        "solver 99 does not exist."
    );
}
