// SPDX-License-Identifier: BSD-3-Clause
//! L1 logistic loss from `solve_l1r_lr`.

#![expect(
    clippy::many_single_char_names,
    reason = "The C solver uses these short names"
)]

use crate::{
    Error, Feature, Parameter,
    l1r_l2_svc::{shrink, transpose},
    l2r_l2_svc::{at, at_mut, same_len, shuffle_prefix, swap_at},
};

#[expect(
    clippy::float_cmp,
    reason = "solve_l1r_lr tests a label of -1 with C equality"
)]
fn negative_label(label: f64) -> bool {
    label == -1.0
}

fn free_bias(index: usize, n: usize, regularize_bias: bool) -> bool {
    n > 0 && index == n.saturating_sub(1) && !regularize_bias
}

fn bias_abs(weights: &[f64]) -> Result<f64, Error> {
    // solve_l1r_lr subtracts fabs of the bias weight from the L1 penalty.
    weights
        .last()
        .map(|value| value.abs())
        .ok_or_else(|| Error::index(0, weights.len()))
}

#[allow(clippy::too_many_arguments, clippy::too_many_lines)] // Keep the source loops together.
#[allow(clippy::manual_midpoint)] // Keep the C addition and multiplication order.
pub(crate) fn solve(
    rows: &[&[Feature]],
    y: &[f64],
    c: &[f64],
    parameter: &Parameter,
    w: &mut [f64],
    eps: f64,
    random: &mut dyn FnMut() -> u32,
    mutated: bool,
) -> Result<(), Error> {
    let l = rows.len();
    let n = w.len();
    same_len(y.len(), l)?;
    same_len(c.len(), l)?;
    let count = f64::from(u32::try_from(l).map_err(|_| Error::Count { count: l })?);
    let columns = transpose(rows, n);
    let mut index: Vec<_> = (0..n).collect();
    let mut hdiag = vec![0.0; n];
    let mut grad = vec![0.0; n];
    let mut wpd = vec![0.0; n];
    let mut xjneg_sum = vec![0.0; n];
    let mut xtd = vec![0.0; l];
    let mut exp_wtx = vec![0.0; l];
    let mut exp_wtx_new = vec![0.0; l];
    let mut tau = vec![0.0; l];
    let mut d = vec![0.0; l];
    let mut inner_eps = 1.0;
    let mut gnorm1_init = -1.0;
    let mut gmax_old = f64::INFINITY;
    w.fill(0.0);
    let mut w_norm = 0.0;
    for (column, ((&weight, wpd_j), xjneg)) in columns
        .iter()
        .zip(w.iter().zip(wpd.iter_mut()).zip(xjneg_sum.iter_mut()))
    {
        w_norm += weight.abs();
        *wpd_j = weight;
        #[expect(
            clippy::indexing_slicing,
            reason = "transpose stores each Problem row index below the shared length of exp_wtx, y and c"
        )]
        {
            for &(i, val) in column {
                exp_wtx[i] += weight * val;
                if negative_label(y[i]) {
                    *xjneg += c[i] * val;
                }
            }
        }
    }
    if !parameter.regularize_bias {
        w_norm -= bias_abs(w)?;
    }
    for (((slot, tau_i), d_i), &cost) in exp_wtx
        .iter_mut()
        .zip(tau.iter_mut())
        .zip(d.iter_mut())
        .zip(c)
    {
        *slot = slot.exp();
        let tau_tmp = 1.0 / (1.0 + *slot);
        *tau_i = cost * tau_tmp;
        *d_i = cost * *slot * tau_tmp * tau_tmp;
    }
    for newton_iter in 0..100 {
        let mut gmax_new = 0.0;
        let mut gnorm1_new = 0.0;
        let mut active_size = n;
        let mut s = 0;
        while s < active_size {
            let j = *at(&index, s)?;
            s = s.checked_add(1).ok_or_else(Error::overflow)?;
            let mut h_j = 1e-12;
            let mut tmp = 0.0;
            let column = at(&columns, j)?;
            #[expect(
                clippy::indexing_slicing,
                reason = "transpose stores each Problem row index below the shared length of d and tau"
            )]
            {
                for &(i, val) in column {
                    h_j += val * val * d[i];
                    tmp += val * tau[i];
                }
            }
            let grad_j = -tmp + *at(&xjneg_sum, j)?;
            *at_mut(&mut hdiag, j)? = h_j;
            *at_mut(&mut grad, j)? = grad_j;
            let weight = *at(w, j)?;
            let mut violation = 0.0;
            if free_bias(j, n, parameter.regularize_bias) {
                violation = grad_j.abs();
            } else {
                let gp = grad_j + 1.0;
                let gn = grad_j - 1.0;
                if weight == 0.0 {
                    if gp < 0.0 {
                        violation = -gp;
                    } else if gn > 0.0 {
                        violation = gn;
                    } else if shrink(gp, gn, gmax_old / count, mutated) {
                        active_size = active_size.checked_sub(1).ok_or_else(Error::overflow)?;
                        s = s.checked_sub(1).ok_or_else(Error::overflow)?;
                        swap_at(&mut index, s, active_size)?;
                        continue;
                    }
                } else if weight > 0.0 {
                    violation = gp.abs();
                } else {
                    violation = gn.abs();
                }
            }
            gmax_new = if gmax_new > violation {
                gmax_new
            } else {
                violation
            };
            gnorm1_new += violation;
        }
        if newton_iter == 0 {
            gnorm1_init = gnorm1_new;
        }
        if gnorm1_new <= eps * gnorm1_init {
            break;
        }
        let mut iter = 0usize;
        let mut qp_gmax_old = f64::INFINITY;
        let mut qp_active_size = active_size;
        xtd.fill(0.0);
        while iter < 1000 {
            let mut qp_gmax_new = 0.0;
            let mut qp_gnorm1_new = 0.0;
            shuffle_prefix(&mut index, qp_active_size, random)?;
            let mut s = 0;
            while s < qp_active_size {
                let j = *at(&index, s)?;
                s = s.checked_add(1).ok_or_else(Error::overflow)?;
                let h = *at(&hdiag, j)?;
                let wpd_j = *at(&wpd, j)?;
                let weight = *at(w, j)?;
                let mut g = *at(&grad, j)? + (wpd_j - weight) * 1e-12;
                let column = at(&columns, j)?;
                #[expect(
                    clippy::indexing_slicing,
                    reason = "transpose stores each Problem row index below the shared length of d and xtd"
                )]
                {
                    for &(i, val) in column {
                        g += val * d[i] * xtd[i];
                    }
                }
                let mut violation = 0.0;
                let mut z;
                if free_bias(j, n, parameter.regularize_bias) {
                    violation = g.abs();
                    z = -g / h;
                } else {
                    let gp = g + 1.0;
                    let gn = g - 1.0;
                    if wpd_j == 0.0 {
                        if gp < 0.0 {
                            violation = -gp;
                        } else if gn > 0.0 {
                            violation = gn;
                        } else if shrink(gp, gn, qp_gmax_old / count, mutated) {
                            qp_active_size =
                                qp_active_size.checked_sub(1).ok_or_else(Error::overflow)?;
                            s = s.checked_sub(1).ok_or_else(Error::overflow)?;
                            swap_at(&mut index, s, qp_active_size)?;
                            continue;
                        }
                    } else if wpd_j > 0.0 {
                        violation = gp.abs();
                    } else {
                        violation = gn.abs();
                    }
                    z = if gp < h * wpd_j {
                        -gp / h
                    } else if gn > h * wpd_j {
                        -gn / h
                    } else {
                        -wpd_j
                    };
                }
                qp_gmax_new = if qp_gmax_new > violation {
                    qp_gmax_new
                } else {
                    violation
                };
                qp_gnorm1_new += violation;
                if z.abs() < 1e-12 {
                    continue;
                }
                z = if z > -10.0 { z } else { -10.0 };
                z = if z < 10.0 { z } else { 10.0 };
                *at_mut(&mut wpd, j)? += z;
                #[expect(
                    clippy::indexing_slicing,
                    reason = "transpose stores each Problem row index below the length of xtd"
                )]
                {
                    for &(i, val) in column {
                        xtd[i] += z * val;
                    }
                }
            }
            iter = iter.checked_add(1).ok_or_else(Error::overflow)?;
            if qp_gnorm1_new <= inner_eps * gnorm1_init {
                if qp_active_size == active_size {
                    break;
                }
                qp_active_size = active_size;
                qp_gmax_old = f64::INFINITY;
                continue;
            }
            qp_gmax_old = qp_gmax_new;
        }
        let mut delta = 0.0;
        let mut w_norm_new = 0.0;
        for ((&grad_j, &wpd_j), &weight) in grad.iter().zip(&wpd).zip(w.iter()) {
            delta += grad_j * (wpd_j - weight);
            if wpd_j != 0.0 {
                w_norm_new += wpd_j.abs();
            }
        }
        if !parameter.regularize_bias {
            w_norm_new -= bias_abs(&wpd)?;
        }
        delta += w_norm_new - w_norm;
        let mut negsum_xtd = 0.0;
        for ((&label, &cost), &step) in y.iter().zip(c).zip(&xtd) {
            if negative_label(label) {
                negsum_xtd += cost * step;
            }
        }
        let mut accepted = false;
        for _ in 0..20 {
            let mut cond = w_norm_new - w_norm + negsum_xtd - 0.01 * delta;
            for (((&step, new_slot), &old), &cost) in xtd
                .iter()
                .zip(exp_wtx_new.iter_mut())
                .zip(exp_wtx.iter())
                .zip(c)
            {
                let exp_xtd = step.exp();
                *new_slot = old * exp_xtd;
                cond += cost * ((1.0 + *new_slot) / (exp_xtd + *new_slot)).ln();
            }
            if cond <= 0.0 {
                w_norm = w_norm_new;
                same_len(w.len(), wpd.len())?;
                w.copy_from_slice(&wpd);
                for ((((slot, &new_exp), tau_i), d_i), &cost) in exp_wtx
                    .iter_mut()
                    .zip(&exp_wtx_new)
                    .zip(tau.iter_mut())
                    .zip(d.iter_mut())
                    .zip(c)
                {
                    *slot = new_exp;
                    let tau_tmp = 1.0 / (1.0 + *slot);
                    *tau_i = cost * tau_tmp;
                    *d_i = cost * *slot * tau_tmp * tau_tmp;
                }
                accepted = true;
                break;
            }
            w_norm_new = 0.0;
            for (wpd_j, &weight) in wpd.iter_mut().zip(w.iter()) {
                *wpd_j = (weight + *wpd_j) * 0.5;
                if *wpd_j != 0.0 {
                    w_norm_new += wpd_j.abs();
                }
            }
            if !parameter.regularize_bias {
                w_norm_new -= bias_abs(&wpd)?;
            }
            delta *= 0.5;
            negsum_xtd *= 0.5;
            for value in &mut xtd {
                *value *= 0.5;
            }
        }
        if !accepted {
            exp_wtx.fill(0.0);
            for (column, &weight) in columns.iter().zip(w.iter()) {
                if weight == 0.0 {
                    continue;
                }
                #[expect(
                    clippy::indexing_slicing,
                    reason = "transpose stores each Problem row index below the length of exp_wtx"
                )]
                {
                    for &(row, val) in column {
                        exp_wtx[row] += weight * val;
                    }
                }
            }
            for value in &mut exp_wtx {
                *value = value.exp();
            }
        }
        if iter == 1 {
            inner_eps *= 0.25;
        }
        gmax_old = gmax_new;
    }
    // The final objective and support count only feed the source print callback.
    Ok(())
}
