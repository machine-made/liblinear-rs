// SPDX-License-Identifier: BSD-3-Clause
//! Dual coordinate descent from `solve_l2r_l1l2_svc`.

#![expect(
    clippy::many_single_char_names,
    reason = "The C solver uses these short names"
)]

use crate::{
    Error, Feature, Parameter, SolverType,
    l2r_l2_svc::{at, at_mut, same_len, shuffle_prefix, sparse_axpy, sparse_dot, swap_at},
};

#[allow(clippy::too_many_arguments, clippy::too_many_lines)] // Keep the source loop in one place.
pub(crate) fn solve(
    rows: &[&[Feature]],
    y: &[f64],
    parameter: &Parameter,
    w: &mut [f64],
    cp: f64,
    cn: f64,
    random: &mut dyn FnMut() -> u32,
) -> Result<usize, Error> {
    let l = rows.len();
    same_len(y.len(), l)?;
    let mut diag = [0.5 / cn, 0.0, 0.5 / cp];
    let mut upper_bound = [f64::INFINITY, 0.0, f64::INFINITY];
    if parameter.solver_type == SolverType::L2rL1lossSvcDual {
        diag = [0.0, 0.0, 0.0];
        upper_bound = [cn, 0.0, cp];
    }
    let mut alpha = vec![0.0; l];
    let mut qd = vec![0.0; l];
    let mut index: Vec<_> = (0..l).collect();
    w.fill(0.0);
    for (i, (row, &yi)) in rows.iter().zip(y).enumerate() {
        let label = if yi > 0.0 { 2 } else { 0 };
        let mut norm = 0.0;
        for feature in *row {
            norm += feature.value * feature.value;
        }
        let diag_yi = *at(&diag, label)?;
        let qd_i = at_mut(&mut qd, i)?;
        *qd_i = diag_yi;
        *qd_i += norm;
        let alpha_i = *at(&alpha, i)?;
        sparse_axpy(yi * alpha_i, row, w);
    }
    let mut active_size = l;
    let mut pgmax_old = f64::INFINITY;
    let mut pgmin_old = f64::NEG_INFINITY;
    let mut iter = 0usize;
    while iter < 300 {
        let mut pgmax_new = f64::NEG_INFINITY;
        let mut pgmin_new = f64::INFINITY;
        shuffle_prefix(&mut index, active_size, random)?;
        let mut s = 0;
        while s < active_size {
            let i = *at(&index, s)?;
            let yi = *at(y, i)?;
            let row = *at(rows, i)?;
            let mut g = yi * sparse_dot(w, row) - 1.0;
            let label = if yi > 0.0 { 2 } else { 0 };
            let bound = *at(&upper_bound, label)?;
            let alpha_i = *at(&alpha, i)?;
            let diag_yi = *at(&diag, label)?;
            g += alpha_i * diag_yi;
            let mut pg = 0.0;
            #[expect(
                clippy::float_cmp,
                reason = "solve_l2r_l1l2_svc compares alpha to 0 and to the C bound"
            )]
            {
                if alpha_i == 0.0 {
                    if g > pgmax_old {
                        active_size = active_size.checked_sub(1).ok_or_else(Error::overflow)?;
                        swap_at(&mut index, s, active_size)?;
                        continue;
                    } else if g < 0.0 {
                        pg = g;
                    }
                } else if alpha_i == bound {
                    if g < pgmin_old {
                        active_size = active_size.checked_sub(1).ok_or_else(Error::overflow)?;
                        swap_at(&mut index, s, active_size)?;
                        continue;
                    } else if g > 0.0 {
                        pg = g;
                    }
                } else {
                    pg = g;
                }
            }
            // Keep the source comparisons, including their NaN and zero rules.
            pgmax_new = if pgmax_new > pg { pgmax_new } else { pg };
            pgmin_new = if pgmin_new < pg { pgmin_new } else { pg };
            if pg.abs() > 1.0e-12 {
                let alpha_old = *at(&alpha, i)?;
                let qd_i = *at(&qd, i)?;
                let next = alpha_old - g / qd_i;
                let lower = if next > 0.0 { next } else { 0.0 };
                *at_mut(&mut alpha, i)? = if lower < bound { lower } else { bound };
                let d = (*at(&alpha, i)? - alpha_old) * yi;
                sparse_axpy(d, row, w);
            }
            s = s.checked_add(1).ok_or_else(Error::overflow)?;
        }
        iter = iter.checked_add(1).ok_or_else(Error::overflow)?;
        if pgmax_new - pgmin_new <= parameter.eps
            && pgmax_new.abs() <= parameter.eps
            && pgmin_new.abs() <= parameter.eps
        {
            if active_size == l {
                break;
            }
            active_size = l;
            pgmax_old = f64::INFINITY;
            pgmin_old = f64::NEG_INFINITY;
            continue;
        }
        pgmax_old = pgmax_new;
        pgmin_old = pgmin_new;
        if pgmax_old <= 0.0 {
            pgmax_old = f64::INFINITY;
        }
        if pgmin_old >= 0.0 {
            pgmin_old = f64::NEG_INFINITY;
        }
    }
    // The source's objective and support count only feed its print callback.
    if parameter.w_recalc {
        w.fill(0.0);
        same_len(alpha.len(), l)?;
        for (row, (&yi, &ai)) in rows.iter().zip(y.iter().zip(&alpha)) {
            if ai > 0.0 {
                sparse_axpy(yi * ai, row, w);
            }
        }
    }
    Ok(iter)
}
