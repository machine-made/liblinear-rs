// SPDX-License-Identifier: BSD-3-Clause
use crate::parity_common::*;
use crate::{SolverType, read_problem, train, train_with_mcsvm_sort};
use serde_json::{Value, json};
use std::{
    collections::BTreeMap,
    fs,
    io::{self, Write},
};

fn changed_sort(values: &mut [f64]) {
    values.sort_by(|a, b| {
        // Change only the source comparison a > b to a < b.
        let changed = a < b;
        if changed {
            std::cmp::Ordering::Less
        } else if a < b {
            std::cmp::Ordering::Greater
        } else {
            std::cmp::Ordering::Equal
        }
    });
}

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
            train_with_mcsvm_sort(&problem, &parameter, changed_sort)
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
fn solver4_matches_every_frozen_case() {
    let n = corpus(SolverType::McsvmCs, false, true).unwrap_or_else(|error| panic!("{error}"));
    assert_eq!(n, 14);
    writeln!(
        io::stdout().lock(),
        "solver 4: {n} of {n} cases equal, worst 0 ULP"
    )
    .unwrap();
}

#[test]
fn changed_sort_is_rejected() {
    let error = corpus(SolverType::McsvmCs, true, false)
        .expect_err("The changed comparison must fail parity.");
    writeln!(io::stdout().lock(), "sort mutation rejected: {error}").unwrap();
}

#[test]
#[ignore = "This check must fail. Run it alone to show the mutation failure."]
fn changed_sort_matches_reference() {
    corpus(SolverType::McsvmCs, true, false).unwrap_or_else(|error| panic!("{error}"));
}

#[test]
fn six_extra_c_problems() {
    let fixtures = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
    let reference = load(&fixtures.join("mcsvm_extra.json"));
    let metadata = load(&fixtures.join("mcsvm_extra.build.json"));
    assert_eq!(
        sha256(&fixtures.join("mcsvm_extra.json")),
        metadata["output_sha256"]
    );
    assert_eq!(
        sha256(&fixtures.join("mcsvm_reference.cpp")),
        metadata["source_sha256"]
    );
    assert_eq!(
        sha256(&root().join("../vendor/liblinear/linear.cpp")),
        metadata["coverage_source_sha256"]
    );
    let rules = load(&root().join("tolerances.json"));
    let cases = reference["problems"].as_array().unwrap();
    assert_eq!(cases.len(), 6);
    for case in cases {
        let name = case["name"].as_str().unwrap();
        let path = fixtures.join(format!("mcsvm_{name}.txt"));
        assert_eq!(sha256(&path), case["input_sha256"]);
        let problem = read_problem(path).unwrap();
        let model = train(&problem, &problem.parameter).unwrap();
        compare(
            &run_json(&model),
            &case["runs"][0],
            name,
            "runs[]",
            &rules["fields"],
            &mut 0,
        )
        .unwrap();
    }
    let sort_metadata = load(&fixtures.join("mcsvm_sort.build.json"));
    assert_eq!(
        sha256(&fixtures.join("mcsvm_sort.json")),
        sort_metadata["output_sha256"]
    );
    assert_eq!(
        sha256(&fixtures.join("mcsvm_sort.c")),
        sort_metadata["source_sha256"]
    );
    writeln!(io::stdout().lock(), "mcsvm extra: 6 of 6 problems equal").unwrap();
}

#[test]
fn one_row_and_binary_layout_and_errors() {
    let problem = read_problem(root().join("corpus/s04_one_row.txt")).unwrap();
    let model = train(&problem, &problem.parameter).unwrap();
    assert_eq!(model.nr_class, 1);
    assert_eq!(model.predictions.len(), 1);
    assert_eq!(model.predictions[0].to_bits(), problem.labels[0].to_bits());
    assert!(model.weights.iter().all(|w| w.to_bits() == 0));
    let problem = read_problem(root().join("corpus/s04_bias_on.txt")).unwrap();
    let model = train(&problem, &problem.parameter).unwrap();
    assert_eq!(model.nr_class, 2);
    assert_eq!(model.weights.len(), 2 * (problem.nr_feature + 1));
    assert!(model.decision_values.iter().all(|row| row.len() == 2));
    let mut parameter = problem.parameter.clone();
    parameter.regularize_bias = false;
    assert!(
        train(&problem, &parameter)
            .unwrap_err()
            .to_string()
            .contains("-R option")
    );
    parameter.regularize_bias = true;
    parameter.init_sol = Some(vec![0.0; model.weights.len()]);
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
}
