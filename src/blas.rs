// SPDX-License-Identifier: BSD-3-Clause
//! The upstream BLAS loops for unit strides.

/// Add `a * x` to `y` in the C loop order.
///
/// # Errors
/// Returns an error when the vector lengths differ.
pub fn daxpy(a: f64, x: &[f64], y: &mut [f64]) -> Result<(), crate::Error> {
    if x.len() != y.len() {
        return Err(crate::Error::Length {
            left: x.len(),
            right: y.len(),
        });
    }
    if a == 0.0 {
        return Ok(());
    }
    // Groups of four match the C unit-stride loop.
    let (y_groups, y_tail) = y.as_chunks_mut::<4>();
    let (x_groups, x_tail) = x.as_chunks::<4>();
    for (slot, entry) in y_groups.iter_mut().zip(x_groups) {
        slot[0] += a * entry[0];
        slot[1] += a * entry[1];
        slot[2] += a * entry[2];
        slot[3] += a * entry[3];
    }
    for (slot, &entry) in y_tail.iter_mut().zip(x_tail) {
        *slot += a * entry;
    }
    Ok(())
}

/// Return the dot product with the C groups of five products.
///
/// # Errors
/// Returns an error when the vector lengths differ.
pub fn ddot(x: &[f64], y: &[f64]) -> Result<f64, crate::Error> {
    if x.len() != y.len() {
        return Err(crate::Error::Length {
            left: x.len(),
            right: y.len(),
        });
    }
    let mut result = 0.0;
    let mut x_rest = x;
    let mut y_rest = y;
    while let Some((x_group, x_next)) = x_rest.split_first_chunk::<5>() {
        let Some((y_group, y_next)) = y_rest.split_first_chunk::<5>() else {
            return Err(crate::Error::Length {
                left: x.len(),
                right: y.len(),
            });
        };
        let &[x0, x1, x2, x3, x4] = x_group;
        let &[y0, y1, y2, y3, y4] = y_group;
        // The five-term sum keeps the C ddot rounding group.
        result += x0 * y0 + x1 * y1 + x2 * y2 + x3 * y3 + x4 * y4;
        x_rest = x_next;
        y_rest = y_next;
    }
    for (&left, &right) in x_rest.iter().zip(y_rest) {
        result += left * right;
    }
    Ok(result)
}

/// Return the scaled norm. The C loop visits the last entry first.
#[must_use]
pub fn dnrm2(x: &[f64]) -> f64 {
    if x.is_empty() {
        return 0.0;
    }
    if let [only] = x {
        return only.abs();
    }
    let mut scale = 0.0;
    let mut ssq = 1.0;
    for &value in x.iter().rev() {
        if value != 0.0 {
            let absxi = value.abs();
            if scale < absxi {
                let temp = scale / absxi;
                ssq = ssq * (temp * temp) + 1.0;
                scale = absxi;
            } else {
                let temp = absxi / scale;
                ssq += temp * temp;
            }
        }
    }
    scale * f64::sqrt(ssq)
}

/// Scale each entry with the C groups of five.
#[allow(clippy::assign_op_pattern)] // Keep the C operand order.
pub fn dscal(a: f64, x: &mut [f64]) {
    let (groups, tail) = x.as_chunks_mut::<5>();
    for group in groups {
        group[0] = a * group[0];
        group[1] = a * group[1];
        group[2] = a * group[2];
        group[3] = a * group[3];
        group[4] = a * group[4];
    }
    for value in tail {
        *value = a * *value;
    }
}
