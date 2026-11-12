// SPDX-License-Identifier: BSD-3-Clause
//! Logistic loss from `l2r_erm_fun` and `l2r_lr_fun`.

use crate::{
    Error, Feature,
    l2r_l2_svc::{same_len, sparse_axpy, sparse_dot, xv},
};

pub(crate) struct Objective<'a> {
    rows: &'a [&'a [Feature]],
    y: &'a [f64],
    c: &'a [f64],
    regularize_bias: bool,
    wx: Vec<f64>,
    tmp: Vec<f64>,
    d: Vec<f64>,
    logarithm: fn(f64) -> f64,
    wtw: f64,
}

impl<'a> Objective<'a> {
    pub(crate) fn new(
        rows: &'a [&'a [Feature]],
        y: &'a [f64],
        c: &'a [f64],
        regularize_bias: bool,
        logarithm: fn(f64) -> f64,
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
            d: vec![0.0; rows.len()],
            logarithm,
            wtw: 0.0,
        })
    }

    fn loss(&self, y: f64, c: f64, wx: f64) -> f64 {
        let ywx = wx * y;
        if ywx >= 0.0 {
            c * (self.logarithm)(1.0 + (-ywx).exp())
        } else {
            c * (-ywx + (self.logarithm)(1.0 + ywx.exp()))
        }
    }

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
            f += self.loss(y, c, wx);
        }
        f + 0.5 * self.wtw
    }

    #[allow(clippy::assign_op_pattern)] // Keep the C operand order.
    fn grad(&mut self, w: &[f64], g: &mut [f64]) {
        for (((tmp, d), (&y, &wx)), &c) in self
            .tmp
            .iter_mut()
            .zip(&mut self.d)
            .zip(self.y.iter().zip(&self.wx))
            .zip(self.c)
        {
            *tmp = 1.0 / (1.0 + (-y * wx).exp());
            *d = *tmp * (1.0 - *tmp);
            *tmp = c * (*tmp - 1.0) * y;
        }
        self.xtv(&self.tmp, g);
        for (slot, &weight) in g.iter_mut().zip(w) {
            *slot = weight + *slot;
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
        for (row, (&c, &d)) in self.rows.iter().zip(self.c.iter().zip(&self.d)) {
            for feature in *row {
                let index = feature.index.wrapping_sub(1);
                #[expect(
                    clippy::indexing_slicing,
                    reason = "Problem checked that each feature index is in 1 through n"
                )]
                {
                    m[index] += feature.value * feature.value * c * d;
                }
            }
        }
    }

    #[allow(clippy::assign_op_pattern)] // Keep the C operand order.
    fn hv(&self, s: &[f64], hs: &mut [f64]) {
        hs.fill(0.0);
        for (row, (&c, &d)) in self.rows.iter().zip(self.c.iter().zip(&self.d)) {
            let mut xts = sparse_dot(s, row);
            xts = c * d * xts;
            sparse_axpy(xts, row, hs);
        }
        for (slot, &step) in hs.iter_mut().zip(s) {
            *slot = step + *slot;
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
                loss += self.loss(y, c, inner_product);
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
