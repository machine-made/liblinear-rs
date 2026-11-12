// SPDX-License-Identifier: BSD-3-Clause
use crate::blas::{daxpy, ddot, dnrm2, dscal};
use serde_json::Value;

fn number(value: &Value) -> f64 {
    f64::from_bits(u64::from_str_radix(value.as_str().unwrap(), 16).unwrap())
}

fn vector(value: &Value) -> Vec<f64> {
    value.as_array().unwrap().iter().map(number).collect()
}

fn check(actual: f64, expected: f64, name: &str) {
    // IEEE arithmetic does not fix the sign or payload of a NaN.
    if expected.is_nan() {
        assert!(actual.is_nan(), "{name}");
        return;
    }
    assert_eq!(
        actual.to_bits(),
        expected.to_bits(),
        "{name}: {actual:?} != {expected:?}"
    );
}

#[test]
fn c_blas_table() {
    let table: Value = serde_json::from_str(include_str!("fixtures/blas_table.json")).unwrap();
    for row in table.as_array().unwrap() {
        let name = row["name"].as_str().unwrap();
        let x = vector(&row["x"]);
        let mut y = vector(&row["y"]);
        let a = number(&row["a"]);
        check(ddot(&x, &y).unwrap(), number(&row["ddot"]), name);
        check(dnrm2(&x), number(&row["dnrm2"]), name);
        daxpy(a, &x, &mut y).unwrap();
        let expected = vector(&row["daxpy"]);
        assert_eq!(y.len(), expected.len(), "{name}");
        for (actual, expected) in y.iter().zip(expected) {
            check(*actual, expected, name);
        }
        let mut scaled = x;
        dscal(a, &mut scaled);
        let expected = vector(&row["dscal"]);
        assert_eq!(scaled.len(), expected.len(), "{name}");
        for (actual, expected) in scaled.iter().zip(expected) {
            check(*actual, expected, name);
        }
    }
}

#[test]
fn unequal_slices_return_length_errors() {
    let x = [1.0, 2.0];
    let mut y = [0.0];
    let message = "Vector lengths 2 and 1 differ. Use two slices of equal length.";
    assert_eq!(ddot(&x, &y).unwrap_err().to_string(), message);
    assert_eq!(daxpy(1.0, &x, &mut y).unwrap_err().to_string(), message);
    assert_eq!(y[0].to_bits(), 0.0_f64.to_bits());
}
