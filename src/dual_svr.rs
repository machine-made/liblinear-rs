// SPDX-License-Identifier: BSD-3-Clause
//! Dual coordinate descent from `solve_l2r_l1l2_svr`.

#![expect(
    clippy::many_single_char_names,
    reason = "The C solver uses these short names"
)]

use crate::{
    Error, Feature, Parameter, SolverType,
    l2r_l2_svc::{at, at_mut, same_len, shuffle_prefix, sparse_axpy, sparse_dot, swap_at},
};

#[allow(clippy::too_many_lines)] // Keep the source loop in one place.
pub(crate) fn solve(
    rows: &[&[Feature]],
    y: &[f64],
    parameter: &Parameter,
    w: &mut [f64],
    random: &mut dyn FnMut() -> u32,
    mutated: bool,
) -> Result<usize, Error> {
    let l = rows.len();
    same_len(y.len(), l)?;
    let p = if mutated {
        parameter.p * 0.5
    } else {
        parameter.p
    };
    let mut lambda = 0.5 / parameter.c;
    let mut upper_bound = f64::INFINITY;
    if parameter.solver_type == SolverType::L2rL1lossSvrDual {
        lambda = 0.0;
        upper_bound = parameter.c;
    }
    let mut beta = vec![0.0; l];
    let mut qd = vec![0.0; l];
    let mut index: Vec<_> = (0..l).collect();
    w.fill(0.0);
    for (i, row) in rows.iter().enumerate() {
        let qd_i = at_mut(&mut qd, i)?;
        for feature in *row {
            *qd_i += feature.value * feature.value;
        }
        let beta_i = *at(&beta, i)?;
        sparse_axpy(beta_i, row, w);
    }
    let mut active_size = l;
    let mut gmax_old = f64::INFINITY;
    let mut gnorm1_init = -1.0;
    let mut iter = 0usize;
    while iter < 300 {
        let mut gmax_new = 0.0;
        let mut gnorm1_new = 0.0;
        shuffle_prefix(&mut index, active_size, random)?;
        let mut s = 0;
        while s < active_size {
            let i = *at(&index, s)?;
            let yi = *at(y, i)?;
            let row = *at(rows, i)?;
            let beta_i = *at(&beta, i)?;
            let mut g = -yi + lambda * beta_i;
            let h = *at(&qd, i)? + lambda;
            g += sparse_dot(w, row);
            let gp = g + p;
            let gn = g - p;
            let mut violation = 0.0;
            if beta_i == 0.0 {
                if gp < 0.0 {
                    violation = -gp;
                } else if gn > 0.0 {
                    violation = gn;
                } else if gp > gmax_old && gn < -gmax_old {
                    active_size = active_size.checked_sub(1).ok_or_else(Error::overflow)?;
                    swap_at(&mut index, s, active_size)?;
                    continue;
                }
            } else if beta_i >= upper_bound {
                if gp > 0.0 {
                    violation = gp;
                } else if gp < -gmax_old {
                    active_size = active_size.checked_sub(1).ok_or_else(Error::overflow)?;
                    swap_at(&mut index, s, active_size)?;
                    continue;
                }
            } else if beta_i <= -upper_bound {
                if gn < 0.0 {
                    violation = -gn;
                } else if gn > gmax_old {
                    active_size = active_size.checked_sub(1).ok_or_else(Error::overflow)?;
                    swap_at(&mut index, s, active_size)?;
                    continue;
                }
            } else if beta_i > 0.0 {
                violation = gp.abs();
            } else {
                violation = gn.abs();
            }
            // Keep the C comparison for NaN and signed zero.
            gmax_new = if gmax_new > violation {
                gmax_new
            } else {
                violation
            };
            gnorm1_new += violation;
            let mut d = if gp < h * beta_i {
                -gp / h
            } else if gn > h * beta_i {
                -gn / h
            } else {
                -beta_i
            };
            if d.abs() < 1.0e-12 {
                s = s.checked_add(1).ok_or_else(Error::overflow)?;
                continue;
            }
            let beta_old = beta_i;
            let next = beta_i + d;
            let lower = if next > -upper_bound {
                next
            } else {
                -upper_bound
            };
            *at_mut(&mut beta, i)? = if lower < upper_bound {
                lower
            } else {
                upper_bound
            };
            d = *at(&beta, i)? - beta_old;
            if d != 0.0 {
                sparse_axpy(d, row, w);
            }
            s = s.checked_add(1).ok_or_else(Error::overflow)?;
        }
        if iter == 0 {
            gnorm1_init = gnorm1_new;
        }
        iter = iter.checked_add(1).ok_or_else(Error::overflow)?;
        if gnorm1_new <= parameter.eps * gnorm1_init {
            if active_size == l {
                break;
            }
            active_size = l;
            gmax_old = f64::INFINITY;
            continue;
        }
        gmax_old = gmax_new;
    }
    // Objective sums and support counts only feed the source print callback.
    Ok(iter)
}
