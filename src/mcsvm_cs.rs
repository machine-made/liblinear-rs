// SPDX-License-Identifier: BSD-3-Clause
//! Joint multiclass descent from `Solver_MCSVM_CS` in `linear.cpp`.

#![expect(
    clippy::many_single_char_names,
    reason = "The C solver uses these short names"
)]

use crate::{
    Error, Feature, Parameter,
    l2r_l2_svc::{at, at_mut, same_len, shuffle_prefix, swap_at},
};
use std::cmp::Ordering;

pub(crate) fn sort(values: &mut [f64]) {
    values.sort_by(|a, b| {
        if a > b {
            Ordering::Less
        } else if a < b {
            Ordering::Greater
        } else {
            Ordering::Equal
        }
    });
}

fn flat_index(feature: usize, class: usize, k: usize, w_len: usize) -> Result<usize, Error> {
    let column = feature
        .checked_sub(1)
        .ok_or_else(|| Error::index(feature, w_len))?;
    let base = column.checked_mul(k).ok_or_else(Error::overflow)?;
    base.checked_add(class).ok_or_else(Error::overflow)
}

fn solve_sub_problem(
    a: f64,
    yi: usize,
    c: f64,
    b: &[f64],
    out: &mut [f64],
    sorter: fn(&mut [f64]),
) -> Result<(), Error> {
    let mut d = b.to_vec();
    if yi < d.len() {
        *at_mut(&mut d, yi)? += a * c;
    }
    sorter(&mut d);
    let first = d.first().copied().ok_or_else(|| Error::index(0, d.len()))?;
    let mut beta = first - a * c;
    let mut r = 1;
    while r < d.len() {
        let value = *at(&d, r)?;
        if beta < count(r)? * value {
            beta += value;
            r = r.checked_add(1).ok_or_else(Error::overflow)?;
        } else {
            break;
        }
    }
    beta /= count(r)?;
    let out_len = out.len();
    let b_len = b.len();
    let slots = out
        .get_mut(..b_len)
        .ok_or_else(|| Error::range(b_len, out_len))?;
    for (r, (slot, &bi)) in slots.iter_mut().zip(b).enumerate() {
        let bound = if r == yi { c } else { 0.0 };
        let value = (beta - bi) / a;
        // The C min template returns its second argument on a tie.
        *slot = if bound < value { bound } else { value };
    }
    Ok(())
}

fn count(n: usize) -> Result<f64, Error> {
    u32::try_from(n)
        .map(f64::from)
        .map_err(|_| Error::Count { count: n })
}

#[expect(
    clippy::float_cmp,
    reason = "Solver_MCSVM_CS::be_shrunk compares alpha with the C bound"
)]
fn at_bound(alpha: f64, bound: f64) -> bool {
    alpha == bound
}

fn be_shrunk(
    m: usize,
    yi: usize,
    alpha: f64,
    min_g: f64,
    g: &[f64],
    c: f64,
) -> Result<bool, Error> {
    let bound = if m == yi { c } else { 0.0 };
    let grad = *at(g, m)?;
    Ok(at_bound(alpha, bound) && grad < min_g)
}

#[allow(clippy::too_many_arguments, clippy::too_many_lines)] // Keep the source loop together.
pub(crate) fn solve(
    rows: &[&[Feature]],
    y: &[usize],
    c: &[f64],
    parameter: &Parameter,
    w: &mut [f64],
    random: &mut dyn FnMut() -> u32,
    sorter: fn(&mut [f64]),
) -> Result<(), Error> {
    let l = rows.len();
    let k = c.len();
    same_len(y.len(), l)?;
    let alpha_len = l.checked_mul(k).ok_or_else(Error::overflow)?;
    let mut alpha = vec![0.0; alpha_len];
    let mut alpha_new = vec![0.0; k];
    let mut index: Vec<_> = (0..l).collect();
    let mut qd = vec![0.0; l];
    let mut d_ind = vec![0; k];
    let mut d_val = vec![0.0; k];
    let mut alpha_index = vec![0; alpha_len];
    let mut y_index = y.to_vec();
    let mut active_size = l;
    let mut active_size_i = vec![k; l];
    let initial = 10.0 * parameter.eps;
    let mut eps_shrink = if initial > 1.0 { initial } else { 1.0 };
    let mut start_from_all = true;
    let mut g = vec![0.0; k];
    let mut b = vec![0.0; k];
    w.fill(0.0);
    for (i, (row, qd_i)) in rows.iter().zip(qd.iter_mut()).enumerate() {
        let start = i.checked_mul(k).ok_or_else(Error::overflow)?;
        let index_len = alpha_index.len();
        let block = alpha_index
            .get_mut(start..)
            .ok_or_else(|| Error::range(start, index_len))?;
        if block.len() < k {
            return Err(Error::range(start.saturating_add(k), index_len));
        }
        for (m, slot) in block.iter_mut().take(k).enumerate() {
            *slot = m;
        }
        for feature in *row {
            *qd_i += feature.value * feature.value;
        }
    }
    for _ in 0..100_000 {
        let mut stopping = f64::NEG_INFINITY;
        shuffle_prefix(&mut index, active_size, random)?;
        let mut s = 0;
        while s < active_size {
            let i = *at(&index, s)?;
            let a = *at(&qd, i)?;
            let start = i.checked_mul(k).ok_or_else(Error::overflow)?;
            let end = start.checked_add(k).ok_or_else(Error::overflow)?;
            let alpha_len = alpha.len();
            let alpha_i = alpha
                .get_mut(start..end)
                .ok_or_else(|| Error::range(start, alpha_len))?;
            let index_len = alpha_index.len();
            let alpha_index_i = alpha_index
                .get_mut(start..end)
                .ok_or_else(|| Error::range(start, index_len))?;
            if a > 0.0 {
                let mut active = *at(&active_size_i, i)?;
                let g_len = g.len();
                g.get_mut(..active)
                    .ok_or_else(|| Error::range(active, g_len))?
                    .fill(1.0);
                let yi = *at(&y_index, i)?;
                if yi < active {
                    *at_mut(&mut g, yi)? = 0.0;
                }
                let row = *at(rows, i)?;
                for feature in row {
                    for m in 0..active {
                        let class = *at(alpha_index_i, m)?;
                        let slot = flat_index(feature.index, class, k, w.len())?;
                        let weight = *at(w, slot)?;
                        *at_mut(&mut g, m)? += weight * feature.value;
                    }
                }
                let mut min_g = f64::INFINITY;
                let mut max_g = f64::NEG_INFINITY;
                for m in 0..active {
                    let class = *at(alpha_index_i, m)?;
                    let alpha_m = *at(alpha_i, class)?;
                    let grad = *at(&g, m)?;
                    if alpha_m < 0.0 && grad < min_g {
                        min_g = grad;
                    }
                    if grad > max_g {
                        max_g = grad;
                    }
                }
                let class = *at(y, i)?;
                let cost = *at(c, class)?;
                let yi = *at(&y_index, i)?;
                if yi < active {
                    let alpha_class = *at(alpha_i, class)?;
                    let g_y = *at(&g, yi)?;
                    if alpha_class < cost && g_y < min_g {
                        min_g = g_y;
                    }
                }
                let mut m = 0;
                while m < active {
                    let class_m = *at(alpha_index_i, m)?;
                    let alpha_m = *at(alpha_i, class_m)?;
                    let yi = *at(&y_index, i)?;
                    if be_shrunk(m, yi, alpha_m, min_g, &g, cost)? {
                        active = active.checked_sub(1).ok_or_else(Error::overflow)?;
                        while active > m {
                            let last = active;
                            let last_class = *at(alpha_index_i, last)?;
                            let last_alpha = *at(alpha_i, last_class)?;
                            if !be_shrunk(last, yi, last_alpha, min_g, &g, cost)? {
                                swap_at(alpha_index_i, m, last)?;
                                swap_at(&mut g, m, last)?;
                                if yi == last {
                                    *at_mut(&mut y_index, i)? = m;
                                } else if yi == m {
                                    *at_mut(&mut y_index, i)? = last;
                                }
                                break;
                            }
                            active = active.checked_sub(1).ok_or_else(Error::overflow)?;
                        }
                    }
                    m = m.checked_add(1).ok_or_else(Error::overflow)?;
                }
                *at_mut(&mut active_size_i, i)? = active;
                if active <= 1 {
                    active_size = active_size.checked_sub(1).ok_or_else(Error::overflow)?;
                    swap_at(&mut index, s, active_size)?;
                    continue;
                }
                if max_g - min_g <= 1e-12 {
                    s = s.checked_add(1).ok_or_else(Error::overflow)?;
                    continue;
                }
                let gap = max_g - min_g;
                stopping = if gap > stopping { gap } else { stopping };
                for m in 0..active {
                    let class_m = *at(alpha_index_i, m)?;
                    let alpha_m = *at(alpha_i, class_m)?;
                    let grad_m = *at(&g, m)?;
                    *at_mut(&mut b, m)? = grad_m - a * alpha_m;
                }
                let yi = *at(&y_index, i)?;
                let b_prefix = b
                    .get(..active)
                    .ok_or_else(|| Error::range(active, b.len()))?;
                solve_sub_problem(a, yi, cost, b_prefix, &mut alpha_new, sorter)?;
                let mut nz_d = 0;
                for m in 0..active {
                    let class_m = *at(alpha_index_i, m)?;
                    let alpha_m = *at(alpha_i, class_m)?;
                    let new_alpha = *at(&alpha_new, m)?;
                    let step = new_alpha - alpha_m;
                    *at_mut(alpha_i, class_m)? = new_alpha;
                    if step.abs() >= 1e-12 {
                        *at_mut(&mut d_ind, nz_d)? = class_m;
                        *at_mut(&mut d_val, nz_d)? = step;
                        nz_d = nz_d.checked_add(1).ok_or_else(Error::overflow)?;
                    }
                }
                let row = *at(rows, i)?;
                for feature in row {
                    for m in 0..nz_d {
                        let class_m = *at(&d_ind, m)?;
                        let step = *at(&d_val, m)?;
                        let slot = flat_index(feature.index, class_m, k, w.len())?;
                        *at_mut(w, slot)? += step * feature.value;
                    }
                }
            }
            s = s.checked_add(1).ok_or_else(Error::overflow)?;
        }
        if stopping < eps_shrink {
            if stopping < parameter.eps && start_from_all {
                break;
            }
            active_size = l;
            active_size_i.fill(k);
            let half = eps_shrink / 2.0;
            eps_shrink = if half > parameter.eps {
                half
            } else {
                parameter.eps
            };
            start_from_all = true;
        } else {
            start_from_all = false;
        }
    }
    // The final objective and support count only feed the C print callback.
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::sort;

    fn palette_bits(palette: &[u64], index: usize) -> Result<u64, String> {
        palette.get(index).copied().ok_or_else(|| {
            format!("Index {index} is outside the sort palette. Use an index below 16.")
        })
    }

    #[test]
    fn sort_matches_ten_thousand_c_arrays() -> Result<(), String> {
        let palette = [
            0,
            0x8000_0000_0000_0000,
            1,
            0x8000_0000_0000_0001,
            0x0010_0000_0000_0000,
            0x8010_0000_0000_0000,
            0x3ff0_0000_0000_0000,
            0xbff0_0000_0000_0000,
            0x4000_0000_0000_0000,
            0xc000_0000_0000_0000,
            0x7fef_ffff_ffff_ffff,
            0xffef_ffff_ffff_ffff,
            0x3fb9_9999_9999_999a,
            0xbfb9_9999_9999_999a,
            0x4010_0000_0000_0000,
            0xc010_0000_0000_0000,
        ];
        let table: serde_json::Value =
            serde_json::from_str(include_str!("../tests/fixtures/mcsvm_sort.json"))
                .map_err(|err| err.to_string())?;
        let seed = table
            .get("seed")
            .and_then(serde_json::Value::as_u64)
            .ok_or_else(|| "The sort table has no seed. Restore the fixture.".to_owned())?;
        let mut state = u32::try_from(seed).map_err(|_| {
            "The sort seed does not fit in 32 bits. Restore the fixture.".to_owned()
        })?;
        let mut next = || -> Result<usize, String> {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let bits = state.checked_shr(16).ok_or_else(|| {
                "The sort seed shift is outside 32 bits. Restore the fixture.".to_owned()
            })?;
            usize::try_from(bits).map_err(|_| {
                "The sort seed does not fit in the index. Restore the fixture.".to_owned()
            })
        };
        let arrays = table
            .get("sorted")
            .and_then(serde_json::Value::as_array)
            .ok_or_else(|| "The sort table has no arrays. Restore the fixture.".to_owned())?;
        assert_eq!(arrays.len(), 10_000);
        for (i, entry) in arrays.iter().enumerate() {
            let span = next()?
                .checked_rem(palette.len())
                .ok_or_else(|| "The sort modulus is zero. Restore the fixture.".to_owned())?;
            let n = 1usize
                .checked_add(span)
                .ok_or_else(|| "The sort length overflowed. Restore the fixture.".to_owned())?;
            let mut values = Vec::with_capacity(n);
            for _ in 0..n {
                let index = next()?
                    .checked_rem(palette.len())
                    .ok_or_else(|| "The sort modulus is zero. Restore the fixture.".to_owned())?;
                values.push(f64::from_bits(palette_bits(&palette, index)?));
            }
            sort(&mut values);
            let text = entry
                .as_str()
                .ok_or_else(|| format!("Array {i} is not text. Restore the sort fixture."))?;
            let mut expected = Vec::with_capacity(text.len());
            for c in text.chars() {
                let digit = c.to_digit(16).ok_or_else(|| {
                    format!("Array {i} has a non-hex digit. Restore the sort fixture.")
                })?;
                let index = usize::try_from(digit).map_err(|_| {
                    format!("Array {i} has a digit outside the index. Restore the sort fixture.")
                })?;
                expected.push(palette_bits(&palette, index)?);
            }
            assert_eq!(
                values.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                expected,
                "array {i}"
            );
        }
        Ok(())
    }
}
