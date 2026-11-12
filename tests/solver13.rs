// SPDX-License-Identifier: BSD-3-Clause
use crate::parity_common::*;
use crate::{SolverType, read_problem, train, train_with_rand};
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
        let mut state = 1_u32;
        let model = if mutated {
            train_with_rand(&problem, &parameter, &mut || {
                state = state.wrapping_mul(1_103_515_245).wrapping_add(12_345);
                (state >> 16) & 0x7fff
            })
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
        if name == "analytic_l2svc" {
            assert_eq!(
                model.weights[0].to_bits(),
                0.999_750_062_484_378_9_f64.to_bits()
            );
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
fn solver13_matches_every_frozen_case() {
    for (solver, expected_count) in [
        (SolverType::L2rL2lossSvcDual, 14),
        (SolverType::L2rL1lossSvcDual, 13),
    ] {
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
fn different_generator_is_rejected() {
    for solver in [SolverType::L2rL2lossSvcDual, SolverType::L2rL1lossSvcDual] {
        let error =
            corpus(solver, true, false).expect_err("The changed generator must fail parity.");
        writeln!(
            io::stdout().lock(),
            "random mutation rejected for solver {}: {error}",
            solver as i32
        )
        .unwrap();
    }
}

#[test]
#[ignore = "This check must fail. Run it alone to show the mutation failure."]
fn different_generator_matches_reference() {
    corpus(SolverType::L2rL2lossSvcDual, true, false).unwrap_or_else(|error| panic!("{error}"));
}

#[test]
fn extra_c_cases_with_and_without_weight_rebuild() {
    let fixtures = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
    let rules = load(&root().join("tolerances.json"));
    let mut rebuild_changes_bits = false;
    for name in ["dual_multiclass", "dual_binary_labels"] {
        let problem = read_problem(fixtures.join(format!("{name}.txt"))).unwrap();
        let original = load(&fixtures.join(format!("{name}.json")));
        for recalc in [false, true] {
            let suffix = if recalc { "_recalc" } else { "" };
            let reference = load(&fixtures.join(format!("{name}{suffix}.json")));
            for (i, &solver) in problem.solvers.iter().enumerate() {
                let mut parameter = problem.parameter.clone();
                parameter.solver_type = solver;
                parameter.w_recalc = recalc;
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
                if recalc {
                    rebuild_changes_bits |= model
                        .weights
                        .iter()
                        .zip(original["runs"][i]["weights"].as_array().unwrap())
                        .any(|(a, b)| a.to_bits() != b.as_f64().unwrap().to_bits());
                }
                if name == "dual_multiclass" {
                    assert_eq!(model.labels, Some(vec![7, -4, 3]));
                } else {
                    assert_eq!(model.labels, Some(vec![1, -1]));
                }
            }
        }
    }
    assert!(
        rebuild_changes_bits,
        "The extra cases must detect an omitted weight rebuild."
    );
}

#[test]
fn dual_parameter_rules() {
    let problem = read_problem(root().join("corpus/s01_bias_on.txt")).unwrap();
    for solver in [SolverType::L2rL2lossSvcDual, SolverType::L2rL1lossSvcDual] {
        let mut parameter = problem.parameter.clone();
        parameter.solver_type = solver;
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
}
