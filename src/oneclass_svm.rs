// SPDX-License-Identifier: BSD-3-Clause
//! One-class descent from `solve_oneclass_svm` in the frozen `linear.cpp`.

#![expect(
    clippy::many_single_char_names,
    reason = "The C solver uses these short names"
)]

use crate::{
    Error, Feature, Parameter,
    l2r_l2_svc::{at, at_mut, sparse_axpy, sparse_dot, swap_at},
};
use std::cmp::Ordering;

type Node = (usize, f64);

#[expect(
    clippy::float_cmp,
    reason = "compare_feature_node breaks a tie with C equality on the value"
)]
fn compare(a: &Node, b: &Node) -> Ordering {
    if a.1 < b.1 {
        Ordering::Less
    } else if a.1 == b.1 {
        a.0.cmp(&b.0)
    } else {
        Ordering::Greater
    }
}

#[expect(
    clippy::float_cmp,
    reason = "solve_oneclass_svm compares alpha with 0 and 1 using C equality"
)]
fn same_alpha(alpha: f64, bound: f64) -> bool {
    alpha == bound
}

fn quick_select(
    nodes: &mut [Node],
    k: usize,
    random: &mut dyn FnMut() -> u32,
) -> Result<(), Error> {
    // C passes high = -1 for an empty range and returns before it draws a pivot.
    if nodes.is_empty() {
        return Ok(());
    }
    let mut low = 0;
    let mut high = nodes.len().checked_sub(1).ok_or_else(Error::overflow)?;
    loop {
        if low == high || high < k {
            return Ok(());
        }
        let span = high.checked_sub(low).ok_or_else(Error::overflow)?;
        let width = span.checked_add(1).ok_or_else(Error::overflow)?;
        let rand = usize::try_from(random()).map_err(|_| Error::overflow())?;
        let offset = rand.checked_rem(width).ok_or_else(Error::overflow)?;
        let pivot = low.checked_add(offset).ok_or_else(Error::overflow)?;
        swap_at(nodes, pivot, high)?;
        let mut index = low;
        for i in low..high {
            let left = *at(nodes, i)?;
            let right = *at(nodes, high)?;
            if compare(&left, &right) == Ordering::Less {
                swap_at(nodes, index, i)?;
                index = index.checked_add(1).ok_or_else(Error::overflow)?;
            }
        }
        swap_at(nodes, high, index)?;
        match index.cmp(&k) {
            Ordering::Equal => return Ok(()),
            Ordering::Greater => {
                high = index.checked_sub(1).ok_or_else(Error::overflow)?;
            }
            Ordering::Less => {
                low = index.checked_add(1).ok_or_else(Error::overflow)?;
            }
        }
    }
}

fn sparse_product(x: &[Feature], y: &[Feature]) -> Result<f64, Error> {
    let mut value = 0.0;
    let (mut i, mut j) = (0, 0);
    while i < x.len() && j < y.len() {
        let left = at(x, i)?;
        let right = at(y, j)?;
        match left.index.cmp(&right.index) {
            Ordering::Equal => {
                value += left.value * right.value;
                i = i.checked_add(1).ok_or_else(Error::overflow)?;
                j = j.checked_add(1).ok_or_else(Error::overflow)?;
            }
            Ordering::Greater => {
                j = j.checked_add(1).ok_or_else(Error::overflow)?;
            }
            Ordering::Less => {
                i = i.checked_add(1).ok_or_else(Error::overflow)?;
            }
        }
    }
    Ok(value)
}

#[allow(
    clippy::too_many_lines,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::manual_midpoint
)]
// Keep the source loop, bounded nu*l conversion and rho sum order.
pub(crate) fn solve(
    rows: &[&[Feature]],
    parameter: &Parameter,
    w: &mut [f64],
    random: &mut dyn FnMut() -> u32,
    mutated: bool,
) -> Result<f64, Error> {
    let l = rows.len();
    let count = f64::from(u32::try_from(l).map_err(|_| Error::Count { count: l })?);
    let n = (parameter.nu * count) as usize;
    let mut alpha = vec![0.0; l];
    let alpha_len = alpha.len();
    alpha
        .get_mut(..n)
        .ok_or_else(|| Error::range(n, alpha_len))?
        .fill(1.0);
    if n < l {
        let fitted = f64::from(u32::try_from(n).map_err(|_| Error::Count { count: n })?);
        *at_mut(&mut alpha, n)? = parameter.nu * count - fitted;
    }
    let mut qd = vec![0.0; l];
    let mut g = vec![0.0; l];
    let mut index: Vec<_> = (0..l).collect();
    let mut active_size = l;
    w.fill(0.0);
    for (row, (qd_i, &alpha_i)) in rows.iter().zip(qd.iter_mut().zip(&alpha)) {
        for feature in *row {
            *qd_i += feature.value * feature.value;
        }
        sparse_axpy(alpha_i, row, w);
    }
    let mut iter = 0usize;
    while iter < 1000 {
        let mut neg_gmax = f64::NEG_INFINITY;
        let mut neg_gmin = f64::INFINITY;
        let active = index
            .get(..active_size)
            .ok_or_else(|| Error::range(active_size, index.len()))?;
        for &i in active {
            let row = *at(rows, i)?;
            let grad = sparse_dot(w, row);
            *at_mut(&mut g, i)? = grad;
            let alpha_i = *at(&alpha, i)?;
            if alpha_i < 1.0 {
                neg_gmax = if neg_gmax > -grad { neg_gmax } else { -grad };
            }
            if alpha_i > 0.0 {
                neg_gmin = if neg_gmin < -grad { neg_gmin } else { -grad };
            }
        }
        if neg_gmax - neg_gmin < parameter.eps {
            if active_size == l {
                break;
            }
            active_size = l;
            continue;
        }
        let mut s = 0;
        while s < active_size {
            let i = *at(&index, s)?;
            let alpha_i = *at(&alpha, i)?;
            let grad = *at(&g, i)?;
            if (same_alpha(alpha_i, 1.0) && -grad > neg_gmax)
                || (same_alpha(alpha_i, 0.0) && -grad < neg_gmin)
            {
                active_size = active_size.checked_sub(1).ok_or_else(Error::overflow)?;
                swap_at(&mut index, s, active_size)?;
            } else {
                s = s.checked_add(1).ok_or_else(Error::overflow)?;
            }
        }
        let mut up = Vec::with_capacity(active_size);
        let mut low = Vec::with_capacity(active_size);
        let active = index
            .get(..active_size)
            .ok_or_else(|| Error::range(active_size, index.len()))?;
        for &i in active {
            let grad = *at(&g, i)?;
            let alpha_i = *at(&alpha, i)?;
            let node = (i, -grad);
            if alpha_i < 1.0 {
                up.push(node);
            }
            if alpha_i > 0.0 {
                low.push(node);
            }
        }
        let tenth = active_size.checked_div(10).ok_or_else(Error::overflow)?;
        let inner = tenth.max(1).min(up.len().min(low.len()));
        let start = up.len().checked_sub(inner).ok_or_else(Error::overflow)?;
        quick_select(&mut up, start, random)?;
        let up_len = up.len();
        up.get_mut(start..)
            .ok_or_else(|| Error::range(start, up_len))?
            .sort_by(compare);
        quick_select(&mut low, inner, random)?;
        let low_len = low.len();
        low.get_mut(..inner)
            .ok_or_else(|| Error::range(inner, low_len))?
            .sort_by(compare);
        for s in 0..inner {
            let pos = up
                .len()
                .checked_sub(s)
                .ok_or_else(Error::overflow)?
                .checked_sub(1)
                .ok_or_else(Error::overflow)?;
            let i = at(&up, pos)?.0;
            let j = at(&low, s)?.0;
            let alpha_i = *at(&alpha, i)?;
            let alpha_j = *at(&alpha, j)?;
            if (same_alpha(alpha_i, 0.0) && same_alpha(alpha_j, 0.0))
                || (same_alpha(alpha_i, 1.0) && same_alpha(alpha_j, 1.0))
            {
                continue;
            }
            let row_i = *at(rows, i)?;
            let row_j = *at(rows, j)?;
            let gi = sparse_dot(w, row_i);
            let gj = sparse_dot(w, row_j);
            let violating = (alpha_i < 1.0 && alpha_j > 0.0 && -gj + 1e-12 < -gi)
                || (alpha_i > 0.0 && alpha_j < 1.0 && -gi + 1e-12 < -gj);
            if !violating {
                continue;
            }
            let qij = sparse_product(row_i, row_j)?;
            let qd_i = *at(&qd, i)?;
            let qd_j = *at(&qd, j)?;
            let mut quad_coef = qd_i + qd_j - 2.0 * qij;
            if quad_coef <= 0.0 {
                quad_coef = 1e-12;
            }
            let mut delta = (gi - gj) / quad_coef;
            let old_alpha_i = *at(&alpha, i)?;
            let sum = old_alpha_i + *at(&alpha, j)?;
            *at_mut(&mut alpha, i)? -= delta;
            *at_mut(&mut alpha, j)? += delta;
            if sum > 1.0 {
                if *at(&alpha, i)? > 1.0 {
                    *at_mut(&mut alpha, i)? = 1.0;
                    *at_mut(&mut alpha, j)? = sum - 1.0;
                }
            } else if *at(&alpha, j)? < 0.0 {
                *at_mut(&mut alpha, j)? = 0.0;
                *at_mut(&mut alpha, i)? = sum;
            }
            if sum > 1.0 {
                if *at(&alpha, j)? > 1.0 {
                    *at_mut(&mut alpha, j)? = 1.0;
                    *at_mut(&mut alpha, i)? = sum - 1.0;
                }
            } else if *at(&alpha, i)? < 0.0 {
                *at_mut(&mut alpha, i)? = 0.0;
                *at_mut(&mut alpha, j)? = sum;
            }
            delta = *at(&alpha, i)? - old_alpha_i;
            sparse_axpy(delta, row_i, w);
            sparse_axpy(-delta, row_j, w);
        }
        iter = iter.checked_add(1).ok_or_else(Error::overflow)?;
    }
    // The objective and support count only feed the C print callback.
    let mut nr_free = 0.0;
    let mut ub = f64::INFINITY;
    let mut lb = f64::NEG_INFINITY;
    let mut sum_free = 0.0;
    for (row, &alpha_i) in rows.iter().zip(&alpha) {
        let grad = sparse_dot(w, row);
        if same_alpha(alpha_i, 1.0) {
            lb = if lb > grad { lb } else { grad };
        } else if same_alpha(alpha_i, 0.0) {
            ub = if ub < grad { ub } else { grad };
        } else {
            nr_free += 1.0;
            sum_free += grad;
        }
    }
    let rho = if nr_free > 0.0 {
        sum_free / nr_free
    } else {
        (ub + lb) / 2.0
    };
    Ok(if mutated { rho + 1.0 } else { rho })
}
