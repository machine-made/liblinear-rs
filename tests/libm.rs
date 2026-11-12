// SPDX-License-Identifier: BSD-3-Clause
//! Check exp and log against the saved C table.
#![expect(
    clippy::indexing_slicing,
    reason = "Tests read known JSON fields and saved rows"
)]
#![expect(
    clippy::unwrap_used,
    reason = "Tests read known files and saved results"
)]
use serde_json::Value;
use std::io::{self, Write};
use std::process::Command;

#[test]
fn matches_c_exp_and_log() {
    let table: Value = serde_json::from_str(include_str!("fixtures/libm_table.json")).unwrap();
    let values = table["values"].as_array().unwrap();
    assert_eq!(values.len(), 2000);
    let mut differences = Vec::new();
    for (i, entry) in values.iter().enumerate() {
        for (input, output, function) in [
            ("exp_input", "exp", f64::exp as fn(f64) -> f64),
            ("log_input", "log", f64::ln as fn(f64) -> f64),
        ] {
            let x = entry[input].as_f64().unwrap();
            let expected = entry[output].as_f64().unwrap();
            let actual = function(x);
            if actual.to_bits() != expected.to_bits() {
                differences.push(format!(
                    "row {i}, {output}({x:.17e}): C {:#018x}, Rust {:#018x}",
                    expected.to_bits(),
                    actual.to_bits()
                ));
            }
        }
    }
    assert!(differences.is_empty(), "{}", differences.join("\n"));
    let version = Command::new("getconf")
        .arg("GNU_LIBC_VERSION")
        .output()
        .unwrap();
    assert!(version.status.success());
    let mut stdout = io::stdout().lock();
    writeln!(
        stdout,
        "libm runner: {}",
        String::from_utf8_lossy(&version.stdout).trim()
    )
    .unwrap();
    writeln!(stdout, "libm: 2000 of 2000 exp and 2000 of 2000 log equal").unwrap();
}
