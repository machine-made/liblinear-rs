// SPDX-License-Identifier: BSD-3-Clause
//! Squared hinge loss from `l2r_erm_fun` and `l2r_l2_svc_fun`.

use crate::{Error, Feature};

pub(crate) fn same_len(left: usize, right: usize) -> Result<(), Error> {
    if left == right {
        Ok(())
    } else {
        Err(Error::Length { left, right })
    }
}

pub(crate) fn at<T>(values: &[T], index: usize) -> Result<&T, Error> {
    values
        .get(index)
        .ok_or_else(|| Error::index(index, values.len()))
}

pub(crate) fn at_mut<T>(values: &mut [T], index: usize) -> Result<&mut T, Error> {
    let length = values.len();
    values
        .get_mut(index)
        .ok_or_else(|| Error::index(index, length))
}

pub(crate) fn swap_at<T>(values: &mut [T], i: usize, j: usize) -> Result<(), Error> {
    let length = values.len();
    if i >= length {
        return Err(Error::index(i, length));
    }
    if j >= length {
        return Err(Error::index(j, length));
    }
    values.swap(i, j);
    Ok(())
}

pub(crate) fn shuffle_prefix(
    index: &mut [usize],
    size: usize,
    random: &mut dyn FnMut() -> u32,
) -> Result<(), Error> {
    if size > index.len() {
        return Err(Error::index(size.saturating_sub(1), index.len()));
    }
    for i in 0..size {
        let rest = size.checked_sub(i).ok_or_else(Error::overflow)?;
        let rand = usize::try_from(random()).map_err(|_| Error::overflow())?;
        let offset = rand.checked_rem(rest).ok_or_else(Error::overflow)?;
        let j = i.checked_add(offset).ok_or_else(Error::overflow)?;
        swap_at(index, i, j)?;
    }
    Ok(())
}

pub(crate) struct Objective<'a> {
    rows: &'a [&'a [Feature]],
    y: &'a [f64],
    c: &'a [f64],
    regularize_bias: bool,
    wx: Vec<f64>,
    tmp: Vec<f64>,
    active: Vec<usize>,
    wtw: f64,
}

#[inline]
pub(crate) fn sparse_dot(w: &[f64], row: &[Feature]) -> f64 {
    let mut value = 0.0;
    for feature in row {
        let index = feature.index.wrapping_sub(1);
        #[expect(
            clippy::indexing_slicing,
            reason = "Problem checked that each feature index is in 1 through n"
        )]
        {
            value += w[index] * feature.value;
        }
    }
    value
}

#[inline]
pub(crate) fn sparse_axpy(a: f64, row: &[Feature], out: &mut [f64]) {
    for feature in row {
        let index = feature.index.wrapping_sub(1);
        #[expect(
            clippy::indexing_slicing,
            reason = "Problem checked that each feature index is in 1 through n"
        )]
        {
            out[index] += a * feature.value;
        }
    }
}

pub(crate) fn xv(rows: &[&[Feature]], v: &[f64], out: &mut [f64]) {
    for (slot, row) in out.iter_mut().zip(rows) {
        *slot = sparse_dot(v, row);
    }
}

impl<'a> Objective<'a> {
    pub(crate) fn new(
        rows: &'a [&'a [Feature]],
        y: &'a [f64],
        c: &'a [f64],
        regularize_bias: bool,
    ) -> Result<Self, Error> {
        same_len(rows.len(), y.len())?;
        same_len(rows.len(), c.len())?;
        Ok(Self {
            rows,
            y,
            c,
            regularize_bias,
            wx: vec![0.0; rows.len()],
            tmp: vec![0.0; rows.len()],
            active: Vec::new(),
            wtw: 0.0,
        })
    }

    fn loss(y: f64, c: f64, wx: f64) -> f64 {
        let d = 1.0 - y * wx;
        if d > 0.0 { c * d * d } else { 0.0 }
    }

    // The row count and the scratch length disagree.
    #[cold]
    #[inline(never)]
    fn reject_row_length(row_count: usize, tmp_count: usize) {
        // Keep the call. Later solver code then stays on the fast boundary.
        std::hint::black_box(row_count.saturating_sub(tmp_count));
    }

    fn sub_xtv(&self, v: &[f64], out: &mut [f64]) {
        out.fill(0.0);
        let n = self.rows.len();
        let Some(rows) = self.rows.get(..n) else {
            return;
        };
        for (&coef, &idx) in v.iter().zip(&self.active) {
            if idx < n {
                #[expect(
                    clippy::indexing_slicing,
                    reason = "the active row index is below the row count"
                )]
                sparse_axpy(coef, rows[idx], out);
            }
        }
    }

    // The parent C class has this full transpose call for other losses.
    #[allow(dead_code)]
    fn xtv(&self, v: &[f64], out: &mut [f64]) {
        out.fill(0.0);
        for (row, &coef) in self.rows.iter().zip(v) {
            sparse_axpy(coef, row, out);
        }
    }
}

impl crate::tron::Objective for Objective<'_> {
    fn fun(&mut self, w: &[f64]) -> f64 {
        self.wtw = 0.0;
        xv(self.rows, w, &mut self.wx);
        for &value in w {
            self.wtw += value * value;
        }
        if !self.regularize_bias
            && let Some(&last) = w.last()
        {
            self.wtw -= last * last;
        }
        let mut f = 0.0;
        for ((&wx, &y), &c) in self.wx.iter().zip(self.y).zip(self.c) {
            f += Self::loss(y, c, wx);
        }
        f + 0.5 * self.wtw
    }

    fn grad(&mut self, w: &[f64], g: &mut [f64]) {
        self.active.clear();
        let row_count = self.rows.len();
        let products = self.wx.get(..row_count);
        let labels = self.y.get(..row_count);
        let penalties = self.c.get(..row_count);
        if let (Some(products), Some(labels), Some(penalties)) = (products, labels, penalties)
            && self.tmp.len() >= row_count
        {
            for i in 0..row_count {
                #[expect(
                    clippy::indexing_slicing,
                    reason = "new checks that products, labels, penalties, and tmp share the row count, and each packed index is below that count"
                )]
                {
                    self.tmp[i] = products[i] * labels[i];
                    if self.tmp[i] < 1.0 {
                        let packed = self.active.len();
                        self.tmp[packed] = penalties[i] * labels[i] * (self.tmp[i] - 1.0);
                        self.active.push(i);
                    }
                }
            }
        } else {
            Self::reject_row_length(row_count, self.tmp.len());
        }
        self.sub_xtv(&self.tmp, g);
        for (slot, &weight) in g.iter_mut().zip(w) {
            *slot = weight + 2.0 * *slot;
        }
        if !self.regularize_bias
            && let (Some(out), Some(&weight)) = (g.last_mut(), w.last())
        {
            *out -= weight;
        }
    }

    fn get_diag_preconditioner(&self, m: &mut [f64]) {
        m.fill(1.0);
        if !self.regularize_bias
            && let Some(last) = m.last_mut()
        {
            *last = 0.0;
        }
        let n = self.rows.len();
        let Some(rows) = self.rows.get(..n) else {
            return;
        };
        let Some(c_all) = self.c.get(..n) else {
            return;
        };
        for &idx in &self.active {
            if idx < n {
                #[expect(
                    clippy::indexing_slicing,
                    reason = "the active row index is below the shared row count"
                )]
                let (row, c) = (rows[idx], c_all[idx]);
                for feature in row {
                    let index = feature.index.wrapping_sub(1);
                    #[expect(
                        clippy::indexing_slicing,
                        reason = "Problem checked that each feature index is in 1 through n"
                    )]
                    {
                        m[index] += feature.value * feature.value * c * 2.0;
                    }
                }
            }
        }
    }

    #[allow(clippy::assign_op_pattern)] // Keep the C operand order.
    fn hv(&self, s: &[f64], hs: &mut [f64]) {
        hs.fill(0.0);
        let n = self.rows.len();
        if let (Some(rows), Some(c_all)) = (self.rows.get(..n), self.c.get(..n)) {
            for &idx in &self.active {
                if idx < n {
                    #[expect(
                        clippy::indexing_slicing,
                        reason = "the active row index is below the shared row count"
                    )]
                    {
                        let row = rows[idx];
                        let mut xts = sparse_dot(s, row);
                        xts = c_all[idx] * xts;
                        sparse_axpy(xts, row, hs);
                    }
                }
            }
        }
        for (slot, &step) in hs.iter_mut().zip(s) {
            *slot = step + 2.0 * *slot;
        }
        if !self.regularize_bias
            && let (Some(out), Some(&step)) = (hs.last_mut(), s.last())
        {
            *out -= step;
        }
    }

    #[allow(clippy::manual_midpoint)] // Keep the C sum and division order.
    fn linesearch_and_update(&mut self, w: &mut [f64], s: &[f64], f: &mut f64, g: &[f64]) -> f64 {
        let mut alpha = 1.0;
        let mut sts = 0.0;
        let mut wts = 0.0;
        let mut gts = 0.0;
        let fold = *f;
        xv(self.rows, s, &mut self.tmp);
        for ((&si, &wi), &gi) in s.iter().zip(w.iter()).zip(g) {
            sts += si * si;
            wts += si * wi;
            gts += si * gi;
        }
        if !self.regularize_bias
            && let (Some(&step), Some(&weight)) = (s.last(), w.last())
        {
            sts -= step * step;
            wts -= step * weight;
        }
        let mut accepted = false;
        for _ in 0..20 {
            let mut loss = 0.0;
            for ((&step, &wx), (&y, &c)) in
                self.tmp.iter().zip(&self.wx).zip(self.y.iter().zip(self.c))
            {
                let inner_product = step * alpha + wx;
                loss += Self::loss(y, c, inner_product);
            }
            *f = loss + (alpha * alpha * sts + self.wtw) / 2.0 + alpha * wts;
            if *f - fold <= 0.01 * alpha * gts {
                for (wx, &step) in self.wx.iter_mut().zip(&self.tmp) {
                    *wx += alpha * step;
                }
                accepted = true;
                break;
            }
            alpha *= 0.5;
        }
        if !accepted {
            *f = fold;
            return 0.0;
        }
        for (slot, &step) in w.iter_mut().zip(s) {
            *slot += alpha * step;
        }
        self.wtw += alpha * alpha * sts + 2.0 * alpha * wts;
        alpha
    }
}
