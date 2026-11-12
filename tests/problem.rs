// SPDX-License-Identifier: BSD-3-Clause
//! Check case-file errors and one-row reads.
#![expect(
    clippy::indexing_slicing,
    reason = "Tests read known JSON fields and saved rows"
)]
#![expect(
    clippy::unwrap_used,
    reason = "Tests read known files and saved results"
)]
use std::{
    fs,
    path::Path,
    sync::atomic::{AtomicUsize, Ordering},
};

use liblinear::{SolverType, read_problem};

static NEXT_FILE: AtomicUsize = AtomicUsize::new(0);

fn with_case(text: &str, check: impl FnOnce(&Path)) {
    let folder = Path::new(env!("CARGO_MANIFEST_DIR")).join("target/parser-tests");
    fs::create_dir_all(&folder).unwrap();
    let path = folder.join(format!(
        "{}-{}.txt",
        std::process::id(),
        NEXT_FILE.fetch_add(1, Ordering::Relaxed)
    ));
    fs::write(&path, text).unwrap();
    check(&path);
    fs::remove_file(path).unwrap();
}

fn case(row: &str) -> String {
    format!(
        "solvers 2\nC 1\neps 0.01\nbias -1\np 0.1\nnu 0.5\nregularize_bias 1\nweights 0\nfeatures 3\nrows 1\ndata\n{row}\n"
    )
}

#[test]
fn missing_file_returns_an_error_with_next_step() {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("target/parser-tests/missing/case.txt");
    let error = read_problem(path).unwrap_err();
    assert!(error.to_string().contains("Check the file and read access"));
    assert!(std::error::Error::source(&error).is_some());
}

#[test]
fn bad_labels_and_features_return_errors_with_line_and_next_step() {
    for row in [
        "bad 1:1",
        "NaN 1:1",
        "inf 1:1",
        "1 0:1",
        "1 2:1 1:1",
        "1 1:1 1:2",
        "1 4:1",
        "1 1:NaN",
        "1 1:2:3",
        "1 1",
        "1 -1:2",
    ] {
        with_case(&case(row), |path| {
            let message = read_problem(path).unwrap_err().to_string();
            assert!(message.starts_with("Line 12:"), "{row}: {message}");
            assert!(
                message.contains("Set ")
                    || message.contains("Use ")
                    || message.contains("Write ")
                    || message.contains("Replace "),
                "{message}"
            );
        });
    }
}

#[test]
fn one_row_case_returns_one_row() {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("reference/corpus/s02_one_row.txt");
    let problem = read_problem(path).unwrap();
    assert_eq!(problem.rows.len(), 1);
    assert_eq!(problem.labels.len(), 1);
}

#[test]
fn bad_headers_and_empty_problems_return_errors() {
    let valid = case("1 1:1");
    for text in [
        valid.replace("rows 1", "rows 0"),
        valid.replace("rows 1", "rows 2"),
        valid.replace("features 3", "features 0"),
        valid.replace("features 3", "features 18446744073709551615"),
        valid.replace("C 1", "C NaN"),
        valid.replace("C 1", "C 1 2"),
        valid.replace("C 1\n", ""),
        valid.replace("C 1", "C 1\nC 1"),
        valid.replace("C 1", "cost 1"),
        valid.replace("data\n", ""),
        valid.replace("data\n", "data 1\n"),
        valid.replace("solvers 2", "solvers"),
        valid.replace("solvers 2", "solvers 8"),
        valid.replace("regularize_bias 1", "regularize_bias 2"),
        valid.replace("weights 0", "weights 1 1 -2"),
        valid.replace("weights 0", "weights 2 1 2"),
        valid.replace("weights 0", "weights 1 1"),
        valid.replace("weights 0", "weights 1 1 inf"),
        valid.replace("1 1:1\n", ""),
    ] {
        with_case(&text, |path| {
            assert_ne!(read_problem(path).unwrap_err().to_string(), "");
        });
    }
}

#[test]
fn bias_weights_and_multiple_solvers_keep_input_order() {
    let text = case("1 1:-0.0 3:2")
        .replace("solvers 2", "solvers 1 2")
        .replace("bias -1", "bias 0")
        .replace("weights 0", "weights 2 -1 2 1 10");
    with_case(&text, |path| {
        let problem = read_problem(path).unwrap();
        assert_eq!(
            problem.solvers,
            [SolverType::L2rL2lossSvcDual, SolverType::L2rL2lossSvc]
        );
        assert_eq!(problem.parameter.solver_type, problem.solvers[0]);
        assert_eq!(problem.parameter.class_weights, [(-1, 2.0), (1, 10.0)]);
        assert_eq!(problem.rows[0].len(), 3);
        assert_eq!(problem.rows[0][0].value.to_bits(), (-0.0_f64).to_bits());
        assert_eq!(problem.rows[0][2].index, 4);
        assert_eq!(problem.rows[0][2].value.to_bits(), 0.0_f64.to_bits());
        assert!(problem.parameter.init_sol.is_none());
        assert!(!problem.parameter.w_recalc);
    });
}

#[test]
fn all_solver_numbers_match_the_reference_enum() {
    for number in [0, 1, 2, 3, 4, 5, 6, 7, 11, 12, 13, 21] {
        assert_eq!(SolverType::try_from(number).unwrap() as i32, number);
    }
    for number in [-1, 8, 10, 14, 20, 22] {
        assert_ne!(SolverType::try_from(number).unwrap_err().to_string(), "");
    }
}
