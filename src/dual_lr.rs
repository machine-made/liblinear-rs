// SPDX-License-Identifier: BSD-3-Clause
//! Dual logistic descent from `solve_l2r_lr_dual`.

#![expect(
    clippy::many_single_char_names,
    reason = "The C solver uses these short names"
)]

use crate::{
    Error, Feature, Parameter,
    l2r_l2_svc::{at, at_mut, same_len, shuffle_prefix, sparse_axpy, sparse_dot},
};

#[allow(clippy::too_many_arguments, clippy::too_many_lines)] // Keep the source loop in one place.
#[allow(clippy::assign_op_pattern)] // Keep the C operand order.
pub(crate) fn solve(
    rows: &[&[Feature]],
    y: &[f64],
    parameter: &Parameter,
    w: &mut [f64],
    cp: f64,
    cn: f64,
    random: &mut dyn FnMut() -> u32,
    logarithm: fn(f64) -> f64,
) -> Result<usize, Error> {
    let l = rows.len();
    same_len(y.len(), l)?;
    let mut xtx = vec![0.0; l];
    let mut index: Vec<_> = (0..l).collect();
    let alpha_len = l.checked_mul(2).ok_or_else(Error::overflow)?;
    let mut alpha = vec![0.0; alpha_len];
    let mut innereps = 1.0e-2;
    let innereps_min = if 1.0e-8 < parameter.eps {
        1.0e-8
    } else {
        parameter.eps
    };
    for (i, &yi) in y.iter().enumerate() {
        let bound = if yi > 0.0 { cp } else { cn };
        let initial = 0.001 * bound;
        let first = if initial < 1.0e-8 { initial } else { 1.0e-8 };
        let even = i.checked_mul(2).ok_or_else(Error::overflow)?;
        let odd = even.checked_add(1).ok_or_else(Error::overflow)?;
        *at_mut(&mut alpha, even)? = first;
        *at_mut(&mut alpha, odd)? = bound - first;
    }
    w.fill(0.0);
    for (i, (row, &yi)) in rows.iter().zip(y).enumerate() {
        let xtx_i = at_mut(&mut xtx, i)?;
        for feature in *row {
            *xtx_i += feature.value * feature.value;
        }
        let even = i.checked_mul(2).ok_or_else(Error::overflow)?;
        let alpha_even = *at(&alpha, even)?;
        sparse_axpy(yi * alpha_even, row, w);
    }
    let mut iter = 0usize;
    while iter < 300 {
        shuffle_prefix(&mut index, l, random)?;
        let mut newton_iter = 0usize;
        let mut gmax = 0.0;
        for &i in &index {
            let yi = *at(y, i)?;
            let row = *at(rows, i)?;
            let bound = if yi > 0.0 { cp } else { cn };
            let a = *at(&xtx, i)?;
            let b = yi * sparse_dot(w, row);
            let even = i.checked_mul(2).ok_or_else(Error::overflow)?;
            let odd = even.checked_add(1).ok_or_else(Error::overflow)?;
            let mut ind1 = even;
            let mut ind2 = odd;
            let mut sign = 1.0;
            if 0.5 * a * (*at(&alpha, ind2)? - *at(&alpha, ind1)?) + b < 0.0 {
                ind1 = odd;
                ind2 = even;
                sign = -1.0;
            }
            let alpha_old = *at(&alpha, ind1)?;
            let mut z = alpha_old;
            if bound - z < 0.5 * bound {
                z = 0.1 * z;
            }
            let mut gp = a * (z - alpha_old) + sign * b + logarithm(z / (bound - z));
            gmax = if gmax > gp.abs() { gmax } else { gp.abs() };
            let mut inner_iter = 0usize;
            while inner_iter <= 100 {
                if gp.abs() < innereps {
                    break;
                }
                let gpp = a + bound / (bound - z) / z;
                let tmpz = z - gp / gpp;
                if tmpz <= 0.0 {
                    z *= 0.1;
                } else {
                    z = tmpz;
                }
                gp = a * (z - alpha_old) + sign * b + logarithm(z / (bound - z));
                newton_iter = newton_iter.checked_add(1).ok_or_else(Error::overflow)?;
                inner_iter = inner_iter.checked_add(1).ok_or_else(Error::overflow)?;
            }
            if inner_iter > 0 {
                *at_mut(&mut alpha, ind1)? = z;
                *at_mut(&mut alpha, ind2)? = bound - z;
                sparse_axpy(sign * (z - alpha_old) * yi, row, w);
            }
        }
        iter = iter.checked_add(1).ok_or_else(Error::overflow)?;
        if gmax < parameter.eps {
            break;
        }
        let tenth = l.checked_div(10).ok_or_else(Error::overflow)?;
        if newton_iter <= tenth {
            let next = 0.1 * innereps;
            innereps = if innereps_min > next {
                innereps_min
            } else {
                next
            };
        }
    }
    // The source's final objective value only feeds its print callback.
    Ok(iter)
}
