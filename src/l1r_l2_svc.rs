// SPDX-License-Identifier: BSD-3-Clause
//! L1 squared hinge loss from `solve_l1r_l2_svc`.

#![expect(
    clippy::many_single_char_names,
    reason = "The C solver uses these short names"
)]

use crate::{
    Error, Feature, Parameter,
    l2r_l2_svc::{same_len, shuffle_prefix, swap_at},
};

pub(crate) fn transpose(rows: &[&[Feature]], n: usize) -> Vec<Vec<(usize, f64)>> {
    let mut columns = vec![Vec::new(); n];
    for (i, row) in rows.iter().enumerate() {
        for feature in *row {
            let column = feature.index.wrapping_sub(1);
            #[expect(
                clippy::indexing_slicing,
                reason = "Problem checked that each feature index is in 1 through n"
            )]
            columns[column].push((i, feature.value));
        }
    }
    columns
}

pub(crate) fn shrink(gp: f64, gn: f64, bound: f64, mutated: bool) -> bool {
    // The mutation changes only the first comparison of the source rule.
    (if mutated { gp < bound } else { gp > bound }) && gn < -bound
}

fn free_bias(index: usize, n: usize, regularize_bias: bool) -> bool {
    n > 0 && index == n.saturating_sub(1) && !regularize_bias
}

#[expect(
    clippy::indexing_slicing,
    reason = "transpose stores each Problem row index below the shared length of b and c"
)]
fn column_curvature(column: &[(usize, f64)], b: &[f64], c: &[f64]) -> (f64, f64) {
    let mut g_loss = 0.0;
    let mut h = 0.0;
    for &(i, val) in column {
        if b[i] > 0.0 {
            let tmp = c[i] * val;
            g_loss -= tmp * b[i];
            h += tmp * val;
        }
    }
    (g_loss, h)
}

#[expect(
    clippy::indexing_slicing,
    reason = "transpose stores each Problem row index below the length of b"
)]
fn add_column(column: &[(usize, f64)], b: &mut [f64], scale: f64) {
    for &(i, val) in column {
        b[i] += scale * val;
    }
}

#[expect(
    clippy::indexing_slicing,
    reason = "transpose stores each Problem row index below the shared length of b and c"
)]
fn column_loss(
    column: &[(usize, f64)],
    b: &mut [f64],
    c: &[f64],
    d_diff: f64,
    track_old: bool,
) -> (f64, f64) {
    let mut loss_old = 0.0;
    let mut loss_new = 0.0;
    for &(i, val) in column {
        if track_old && b[i] > 0.0 {
            loss_old += c[i] * b[i] * b[i];
        }
        let b_new = b[i] + d_diff * val;
        b[i] = b_new;
        if b_new > 0.0 {
            loss_new += c[i] * b_new * b_new;
        }
    }
    (loss_old, loss_new)
}

#[allow(clippy::too_many_arguments, clippy::too_many_lines)] // Keep the source loop together.
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
    let mut columns = transpose(rows, n);
    let mut index: Vec<_> = (0..n).collect();
    let mut b = vec![1.0; l];
    let mut xj_sq = vec![0.0; n];
    w.fill(0.0);
    for (column, (&weight, xj)) in columns.iter_mut().zip(w.iter().zip(xj_sq.iter_mut())) {
        #[expect(
            clippy::indexing_slicing,
            reason = "transpose stores each Problem row index below the shared length of y, b and c"
        )]
        {
            for (i, val) in &mut *column {
                *val *= y[*i];
                b[*i] -= weight * *val;
                *xj += c[*i] * *val * *val;
            }
        }
    }
    let mut active_size = n;
    let mut gmax_old = f64::INFINITY;
    let mut gnorm1_init = -1.0;
    let mut loss_old = 0.0;
    for iter in 0..1000 {
        let mut gmax_new = 0.0;
        let mut gnorm1_new = 0.0;
        shuffle_prefix(&mut index, active_size, random)?;
        let mut s = 0;
        while s < active_size {
            // index starts as 0..n and swap keeps every entry in that range.
            #[expect(
                clippy::indexing_slicing,
                reason = "index is a permutation of the weight length, and transpose stores each Problem row index below b and c"
            )]
            {
                let j = index[s];
                s = s.checked_add(1).ok_or_else(Error::overflow)?;
                let (mut g_loss, mut h) = column_curvature(&columns[j], &b, c);
                g_loss *= 2.0;
                let g = g_loss;
                h *= 2.0;
                h = if h > 1e-12 { h } else { 1e-12 };
                let weight = w[j];
                let bias_free = free_bias(j, n, parameter.regularize_bias);
                let mut violation = 0.0;
                let mut gp = 0.0;
                let mut gn = 0.0;
                if bias_free {
                    violation = g.abs();
                } else {
                    gp = g + 1.0;
                    gn = g - 1.0;
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
                let mut d = if bias_free {
                    -g / h
                } else if gp < h * weight {
                    -gp / h
                } else if gn > h * weight {
                    -gn / h
                } else {
                    -weight
                };
                if d.abs() < 1e-12 {
                    continue;
                }
                let mut delta = if bias_free {
                    g * d
                } else {
                    (weight + d).abs() - weight.abs() + g * d
                };
                let mut d_old = 0.0;
                let mut accepted = false;
                let xj = xj_sq[j];
                for num_linesearch in 0..20 {
                    let d_diff = d_old - d;
                    let mut cond = if bias_free {
                        -0.01 * delta
                    } else {
                        (weight + d).abs() - weight.abs() - 0.01 * delta
                    };
                    let appxcond = xj * d * d + g_loss * d + cond;
                    if appxcond <= 0.0 {
                        add_column(&columns[j], &mut b, d_diff);
                        accepted = true;
                        break;
                    }
                    let loss_new = if num_linesearch == 0 {
                        let (old, new) = column_loss(&columns[j], &mut b, c, d_diff, true);
                        loss_old = old;
                        new
                    } else {
                        column_loss(&columns[j], &mut b, c, d_diff, false).1
                    };
                    cond = cond + loss_new - loss_old;
                    if cond <= 0.0 {
                        accepted = true;
                        break;
                    }
                    d_old = d;
                    d *= 0.5;
                    delta *= 0.5;
                }
                w[j] += d;
                if !accepted {
                    b.fill(1.0);
                    for (column, &weight) in columns.iter().zip(w.iter()) {
                        if weight == 0.0 {
                            continue;
                        }
                        add_column(column, &mut b, -weight);
                    }
                }
            }
        }
        if iter == 0 {
            gnorm1_init = gnorm1_new;
        }
        if gnorm1_new <= eps * gnorm1_init {
            if active_size == n {
                break;
            }
            active_size = n;
            gmax_old = f64::INFINITY;
            continue;
        }
        gmax_old = gmax_new;
    }
    // The source restores its private columns and computes print-only values here.
    Ok(())
}
