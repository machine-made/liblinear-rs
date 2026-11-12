// SPDX-License-Identifier: BSD-3-Clause
use crate::Model;
use serde_json::{Value, json};
use std::{
    collections::BTreeMap,
    fs,
    io::Write,
    path::{Path, PathBuf},
    process::{Command, Stdio},
};

pub(crate) fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("reference")
}

pub(crate) fn load(path: &Path) -> Value {
    serde_json::from_slice(&fs::read(path).unwrap()).unwrap()
}

pub(crate) fn sha256(path: &Path) -> String {
    let output = Command::new("sha256sum").arg(path).output().unwrap();
    assert!(output.status.success());
    String::from_utf8(output.stdout)
        .unwrap()
        .split_whitespace()
        .next()
        .unwrap()
        .to_owned()
}

pub(crate) fn candidate_sha256(value: &Value) -> String {
    let mut child = Command::new("sha256sum")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .unwrap();
    child
        .stdin
        .take()
        .unwrap()
        .write_all(&serde_json::to_vec(value).unwrap())
        .unwrap();
    let output = child.wait_with_output().unwrap();
    assert!(output.status.success());
    String::from_utf8(output.stdout)
        .unwrap()
        .split_whitespace()
        .next()
        .unwrap()
        .to_owned()
}

pub(crate) fn field_counts(value: &Value, path: &str, counts: &mut BTreeMap<String, usize>) {
    match value {
        Value::Number(_) => {
            *counts.entry(path.to_owned()).or_default() += 1;
        }
        Value::Array(values) => {
            for v in values {
                field_counts(v, &format!("{path}[]"), counts);
            }
        }
        Value::Object(values) => {
            for (key, v) in values {
                field_counts(v, &format!("{path}.{key}"), counts);
            }
        }
        _ => {}
    }
}

pub(crate) fn run_json(model: &Model) -> Value {
    json!({"solver": model.parameter.solver_type as i32, "C": model.parameter.c,
        "eps": model.parameter.eps, "bias": model.bias, "nr_class": model.nr_class,
        "nr_feature": model.nr_feature, "rho": model.rho, "labels": model.labels,
        "weights": model.weights, "decision_values": model.decision_values,
        "predictions": model.predictions})
}

pub(crate) fn ordered_bits(value: f64) -> u64 {
    let bits = value.to_bits();
    if bits >> 63 == 0 {
        bits | (1 << 63)
    } else {
        !bits
    }
}

pub(crate) fn compare(
    actual: &Value,
    expected: &Value,
    path: &str,
    rule_path: &str,
    rules: &Value,
    count: &mut usize,
) -> Result<(), String> {
    if !rule_path.is_empty() {
        let rule = &rules[rule_path];
        assert_eq!(rule["kind"], "exact", "{rule_path}");
        assert_eq!(rule["value"], 0, "{rule_path}");
        assert_ne!(rule["reason"].as_str().unwrap(), "");
    }
    match (actual, expected) {
        (Value::Number(a), Value::Number(b)) => {
            *count += 1;
            let a = a.as_f64().unwrap();
            let b = b.as_f64().unwrap();
            if !a.is_finite() || !b.is_finite() || a.to_bits() != b.to_bits() {
                return Err(format!(
                    "{path}: got {a:.17e} (0x{:016x}), expected {b:.17e} (0x{:016x}), {} ULP",
                    a.to_bits(),
                    b.to_bits(),
                    ordered_bits(a).abs_diff(ordered_bits(b))
                ));
            }
        }
        (Value::Array(a), Value::Array(b)) if a.len() == b.len() => {
            for (i, (a, b)) in a.iter().zip(b).enumerate() {
                compare(
                    a,
                    b,
                    &format!("{path}[{i}]"),
                    &format!("{rule_path}[]"),
                    rules,
                    count,
                )?;
            }
        }
        (Value::Object(a), Value::Object(b)) if a.keys().eq(b.keys()) => {
            for (key, a) in a {
                let rule = if rule_path.is_empty() {
                    key.clone()
                } else {
                    format!("{rule_path}.{key}")
                };
                compare(a, &b[key], &format!("{path}.{key}"), &rule, rules, count)?;
            }
        }
        _ if actual == expected => {}
        _ => return Err(format!("{path}: unequal value or shape")),
    }
    Ok(())
}
