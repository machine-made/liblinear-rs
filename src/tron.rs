// SPDX-License-Identifier: BSD-3-Clause
//! LIBLINEAR 2.50 uses NEWTON with line search, not the older TRON method.

#![expect(
    clippy::many_single_char_names,
    reason = "The C solver uses these short names"
)]

use crate::{
    Error,
    blas::{daxpy, dnrm2, dscal},
};

pub(crate) trait Objective {
    fn fun(&mut self, w: &[f64]) -> f64;
    fn grad(&mut self, w: &[f64], g: &mut [f64]);
    fn get_diag_preconditioner(&self, m: &mut [f64]);
    fn hv(&self, s: &[f64], hs: &mut [f64]);
    fn linesearch_and_update(&mut self, w: &mut [f64], s: &[f64], f: &mut f64, g: &[f64]) -> f64;
}

pub(crate) type Dot = fn(&[f64], &[f64]) -> Result<f64, Error>;

pub(crate) fn newton(
    objective: &mut impl Objective,
    w: &mut [f64],
    eps: f64,
    dot: Dot,
) -> Result<(), Error> {
    let n = w.len();
    let mut s = vec![0.0; n];
    let mut r = vec![0.0; n];
    let mut g = vec![0.0; n];
    let mut m = vec![0.0; n];
    let w0 = vec![0.0; n];
    objective.fun(&w0);
    objective.grad(&w0, &mut g);
    let gnorm0 = dnrm2(&g);
    let mut f = objective.fun(w);
    objective.grad(w, &mut g);
    let gnorm = dnrm2(&g);
    if gnorm <= eps * gnorm0 {
        return Ok(());
    }
    for _ in 1..=1000 {
        objective.get_diag_preconditioner(&mut m);
        for value in &mut m {
            *value = (1.0 - 0.01) + 0.01 * *value;
        }
        pcg(objective, &g, &m, &mut s, &mut r, dot)?;
        let fold = f;
        let step_size = objective.linesearch_and_update(w, &s, &mut f, &g);
        if step_size == 0.0 {
            break;
        }
        objective.grad(w, &mut g);
        let gnorm = dnrm2(&g);
        if gnorm <= eps * gnorm0 {
            break;
        }
        if f < -1.0e32 {
            break;
        }
        let actred = fold - f;
        if actred.abs() <= 1.0e-12 * f.abs() {
            break;
        }
    }
    Ok(())
}

fn pcg(
    objective: &impl Objective,
    g: &[f64],
    m: &[f64],
    s: &mut [f64],
    r: &mut [f64],
    dot: Dot,
) -> Result<(), Error> {
    let n = g.len();
    same_len(n, m.len())?;
    same_len(n, s.len())?;
    same_len(n, r.len())?;
    let mut d = vec![0.0; n];
    let mut hd = vec![0.0; n];
    let mut z = vec![0.0; n];
    let mut q = 0.0;
    for ((((slot_s, slot_r), &gradient), &precond), (slot_z, slot_d)) in s
        .iter_mut()
        .zip(r.iter_mut())
        .zip(g)
        .zip(m)
        .zip(z.iter_mut().zip(d.iter_mut()))
    {
        *slot_s = 0.0;
        *slot_r = -gradient;
        *slot_z = *slot_r / precond;
        *slot_d = *slot_z;
    }
    let mut ztr = dot(&z, r)?;
    let g_minv_norm = ztr.sqrt();
    let root = g_minv_norm.sqrt();
    let cgtol = if 0.5 < root { 0.5 } else { root };
    for cg_iter in 1..=n.max(5) {
        objective.hv(&d, &mut hd);
        let dhd = dot(&d, &hd)?;
        if dhd <= 1.0e-16 {
            break;
        }
        let mut alpha = ztr / dhd;
        daxpy(alpha, &d, s)?;
        alpha = -alpha;
        daxpy(alpha, &hd, r)?;
        let residual_dot = dot(s, r)?;
        let gradient_dot = dot(s, g)?;
        let newq = -0.5 * (residual_dot - gradient_dot);
        let qdiff = newq - q;
        if newq <= 0.0 && qdiff <= 0.0 {
            let cg_iter =
                f64::from(u32::try_from(cg_iter).map_err(|_| Error::Count { count: cg_iter })?);
            if cg_iter * qdiff >= cgtol * newq {
                break;
            }
        } else {
            break;
        }
        q = newq;
        for (slot_z, (&residual, &precond)) in z.iter_mut().zip(r.iter().zip(m)) {
            *slot_z = residual / precond;
        }
        let znewtrnew = dot(&z, r)?;
        let beta = znewtrnew / ztr;
        dscal(beta, &mut d);
        daxpy(1.0, &z, &mut d)?;
        ztr = znewtrnew;
    }
    Ok(())
}

fn same_len(left: usize, right: usize) -> Result<(), Error> {
    if left == right {
        Ok(())
    } else {
        Err(Error::Length { left, right })
    }
}
