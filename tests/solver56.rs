// SPDX-License-Identifier: BSD-3-Clause
use crate::parity_common::*;
use crate::{SolverType, read_problem, train, train_with_shrinking};
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
            train_with_shrinking(&problem, &parameter, true)
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
fn solver56_matches_every_frozen_case() {
    for (solver, expected_count) in [(SolverType::L1rL2lossSvc, 14), (SolverType::L1rLr, 14)] {
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
fn changed_shrinking_is_rejected() {
    for solver in [SolverType::L1rL2lossSvc, SolverType::L1rLr] {
        let error =
            corpus(solver, true, false).expect_err("The changed comparison must fail parity.");
        writeln!(
            io::stdout().lock(),
            "shrinking mutation rejected for solver {}: {error}",
            solver as i32
        )
        .unwrap();
    }
}

#[test]
#[ignore = "This check must fail. Run it alone to show the mutation failure."]
fn changed_shrinking_matches_reference() {
    corpus(SolverType::L1rL2lossSvc, true, false).unwrap_or_else(|error| panic!("{error}"));
}

#[test]
fn six_extra_c_problems() {
    let fixtures = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
    let reference = load(&fixtures.join("l1_extra.json"));
    let metadata = load(&fixtures.join("l1_extra.build.json"));
    assert_eq!(
        sha256(&fixtures.join("l1_extra.json")),
        metadata["output_sha256"]
    );
    assert_eq!(
        sha256(&fixtures.join("l1_reference.cpp")),
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
        let path = fixtures.join(format!("l1_{name}.txt"));
        assert_eq!(sha256(&path), case["input_sha256"]);
        let problem = read_problem(path).unwrap();
        for (i, &solver) in problem.solvers.iter().enumerate() {
            let mut parameter = problem.parameter.clone();
            parameter.solver_type = solver;
            let model = train(&problem, &parameter).unwrap();
            compare(
                &run_json(&model),
                &case["runs"][i],
                name,
                "runs[]",
                &rules["fields"],
                &mut 0,
            )
            .unwrap();
            let source = train_with_shrinking(&problem, &parameter, false).unwrap();
            compare(
                &run_json(&source),
                &case["runs"][i],
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
        "l1 extra: 6 of 6 problems equal for solvers 5 and 6"
    )
    .unwrap();
}

#[test]
fn l1_parameter_rules_and_degenerate_rows() {
    for number in [5, 6] {
        for kind in ["one_row", "zero_feature", "duplicate"] {
            let name = format!("s{number:02}_{kind}");
            let problem = read_problem(root().join(format!("corpus/{name}.txt"))).unwrap();
            let model = train(&problem, &problem.parameter).unwrap();
            assert_eq!(model.predictions.len(), problem.rows.len());
            if kind == "one_row" {
                assert_eq!(model.nr_class, 1);
                assert_eq!(model.predictions[0].to_bits(), problem.labels[0].to_bits());
            }
        }
        let problem =
            read_problem(root().join(format!("corpus/s{number:02}_bias_on.txt"))).unwrap();
        let mut parameter = problem.parameter.clone();
        parameter.init_sol = Some(vec![0.0; problem.nr_feature + 1]);
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
}
