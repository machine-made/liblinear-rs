// SPDX-License-Identifier: BSD-3-Clause
//! Case file types and the case reader.
//!
//! ```
//! use liblinear::Feature;
//! let feature = Feature { index: 1, value: 0.5 };
//! assert_eq!(feature.index, 1);
//! ```

use std::{collections::BTreeMap, fs, path::Path, str::FromStr};

use crate::{Error, Parameter, SolverType};

/// One sparse feature. Indices start at 1.
///
/// ```
/// use liblinear::Feature;
/// let feature = Feature { index: 1, value: -0.5 };
/// assert_eq!(feature.value.to_bits(), (-0.5_f64).to_bits());
/// ```
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Feature {
    /// Feature index. The first feature is 1.
    ///
    /// ```
    /// let feature = liblinear::Feature { index: 2, value: 1.0 };
    /// assert_eq!(feature.index, 2);
    /// ```
    pub index: usize,
    /// Feature value supplied by the caller.
    ///
    /// ```
    /// let feature = liblinear::Feature { index: 1, value: 0.25 };
    /// assert_eq!(feature.value.to_bits(), 0.25_f64.to_bits());
    /// ```
    pub value: f64,
}

/// One case file, with the requested solvers and shared training inputs.
///
/// ```
/// use liblinear::{Feature, Parameter, Problem, SolverType};
/// let parameter = Parameter {
///     solver_type: SolverType::L2rL2lossSvc,
///     eps: 0.1,
///     c: 1.0,
///     class_weights: Vec::new(),
///     p: 0.1,
///     nu: 0.5,
///     init_sol: None,
///     regularize_bias: true,
///     w_recalc: false,
/// };
/// let problem = Problem {
///     solvers: vec![SolverType::L2rL2lossSvc],
///     parameter: parameter.clone(),
///     nr_feature: 1,
///     bias: -1.0,
///     labels: vec![1.0],
///     rows: vec![vec![Feature { index: 1, value: 1.0 }]],
/// };
/// assert_eq!(problem.rows.len(), 1);
/// ```
#[derive(Clone, Debug, PartialEq)]
pub struct Problem {
    /// Solver numbers in case order.
    ///
    /// ```
    /// # use liblinear::{Feature, Parameter, Problem, SolverType};
    /// # let parameter = Parameter { solver_type: SolverType::L2rL2lossSvc, eps: 0.1, c: 1.0, class_weights: Vec::new(), p: 0.1, nu: 0.5, init_sol: None, regularize_bias: true, w_recalc: false };
    /// # let problem = Problem { solvers: vec![SolverType::L2rL2lossSvc], parameter, nr_feature: 1, bias: -1.0, labels: vec![1.0], rows: vec![vec![Feature { index: 1, value: 1.0 }]] };
    /// assert_eq!(problem.solvers[0] as i32, 2);
    /// ```
    pub solvers: Vec<SolverType>,
    /// The solver is the first entry in `solvers`. Select each solver to train.
    ///
    /// ```
    /// # use liblinear::{Feature, Parameter, Problem, SolverType};
    /// # let parameter = Parameter { solver_type: SolverType::L2rL2lossSvc, eps: 0.1, c: 1.0, class_weights: Vec::new(), p: 0.1, nu: 0.5, init_sol: None, regularize_bias: true, w_recalc: false };
    /// # let problem = Problem { solvers: vec![SolverType::L2rL2lossSvc], parameter: parameter.clone(), nr_feature: 1, bias: -1.0, labels: vec![1.0], rows: vec![vec![Feature { index: 1, value: 1.0 }]] };
    /// assert_eq!(problem.parameter.solver_type as i32, 2);
    /// ```
    pub parameter: Parameter,
    /// Feature count before the bias feature, including unused features.
    ///
    /// ```
    /// # use liblinear::{Feature, Parameter, Problem, SolverType};
    /// # let parameter = Parameter { solver_type: SolverType::L2rL2lossSvc, eps: 0.1, c: 1.0, class_weights: Vec::new(), p: 0.1, nu: 0.5, init_sol: None, regularize_bias: true, w_recalc: false };
    /// # let problem = Problem { solvers: vec![SolverType::L2rL2lossSvc], parameter, nr_feature: 3, bias: -1.0, labels: vec![1.0], rows: vec![vec![Feature { index: 1, value: 1.0 }]] };
    /// assert_eq!(problem.nr_feature, 3);
    /// ```
    pub nr_feature: usize,
    /// Bias value. A negative value omits the bias feature.
    ///
    /// ```
    /// # use liblinear::{Feature, Parameter, Problem, SolverType};
    /// # let parameter = Parameter { solver_type: SolverType::L2rL2lossSvc, eps: 0.1, c: 1.0, class_weights: Vec::new(), p: 0.1, nu: 0.5, init_sol: None, regularize_bias: true, w_recalc: false };
    /// # let problem = Problem { solvers: vec![SolverType::L2rL2lossSvc], parameter, nr_feature: 1, bias: -1.0, labels: vec![1.0], rows: vec![vec![Feature { index: 1, value: 1.0 }]] };
    /// assert!(problem.bias < 0.0);
    /// ```
    pub bias: f64,
    /// One label or target for each row.
    ///
    /// ```
    /// # use liblinear::{Feature, Parameter, Problem, SolverType};
    /// # let parameter = Parameter { solver_type: SolverType::L2rL2lossSvc, eps: 0.1, c: 1.0, class_weights: Vec::new(), p: 0.1, nu: 0.5, init_sol: None, regularize_bias: true, w_recalc: false };
    /// # let problem = Problem { solvers: vec![SolverType::L2rL2lossSvc], parameter, nr_feature: 1, bias: -1.0, labels: vec![1.0], rows: vec![vec![Feature { index: 1, value: 1.0 }]] };
    /// assert_eq!(problem.labels.len(), 1);
    /// ```
    pub labels: Vec<f64>,
    /// Each row includes the bias feature when bias is nonnegative.
    ///
    /// Vector lengths replace the C end markers.
    ///
    /// ```
    /// # use liblinear::{Feature, Parameter, Problem, SolverType};
    /// # let parameter = Parameter { solver_type: SolverType::L2rL2lossSvc, eps: 0.1, c: 1.0, class_weights: Vec::new(), p: 0.1, nu: 0.5, init_sol: None, regularize_bias: true, w_recalc: false };
    /// # let problem = Problem { solvers: vec![SolverType::L2rL2lossSvc], parameter, nr_feature: 1, bias: -1.0, labels: vec![1.0], rows: vec![vec![Feature { index: 1, value: 1.0 }]] };
    /// assert_eq!(problem.rows[0][0].index, 1);
    /// ```
    pub rows: Vec<Vec<Feature>>,
}

/// Read the case format in `reference/README.md`.
///
/// # Errors
///
/// Returns an error if the file cannot be read, a header field is missing or
/// invalid, the row count is wrong, or a row has an invalid label or feature.
///
/// ```
/// let path = std::env::temp_dir().join("liblinear_doc_case.txt");
/// std::fs::write(
///     &path,
///     "solvers 2\nC 1\neps 0.1\nbias -1\np 0.1\nnu 0.5\nregularize_bias 1\nweights 0\nfeatures 1\nrows 1\ndata\n1 1:1\n",
/// )?;
/// let problem = liblinear::read_problem(&path)?;
/// assert_eq!(problem.nr_feature, 1);
/// # Ok::<(), liblinear::Error>(())
/// ```
pub fn read_problem(path: impl AsRef<Path>) -> Result<Problem, Error> {
    parse_problem(&fs::read_to_string(path)?)
}

type Header<'a> = BTreeMap<&'a str, (usize, Vec<&'a str>)>;

fn parse_problem(text: &str) -> Result<Problem, Error> {
    let mut lines = text.lines().enumerate();
    let mut header = Header::new();
    let mut has_data = false;
    for (offset, line) in lines.by_ref() {
        let mut fields = line.split_whitespace();
        let Some(key) = fields.next() else { continue };
        let line_number = next_line(offset)?;
        if key == "data" {
            if fields.next().is_some() {
                return Err(Error::input(line_number, "Put data on a line by itself."));
            }
            has_data = true;
            break;
        }
        if !matches!(
            key,
            "solvers"
                | "C"
                | "eps"
                | "bias"
                | "p"
                | "nu"
                | "regularize_bias"
                | "weights"
                | "features"
                | "rows"
        ) {
            return Err(Error::input(
                line_number,
                "Use a header key from the case format.",
            ));
        }
        if header
            .insert(key, (line_number, fields.collect()))
            .is_some()
        {
            return Err(Error::input(line_number, "Remove the repeated header key."));
        }
    }
    if !has_data {
        return Err(Error::input(
            0,
            "Add the header and a data line before the rows.",
        ));
    }
    let (line, values) = field(&header, "solvers")?;
    let solvers: Vec<SolverType> = values
        .iter()
        .map(|value| {
            SolverType::try_from(number::<i32>(value, line)?).map_err(|_| {
                Error::input(line, "Use solver 0, 1, 2, 3, 4, 5, 6, 7, 11, 12, 13 or 21.")
            })
        })
        .collect::<Result<_, _>>()?;
    let Some(&solver_type) = solvers.first() else {
        return Err(Error::input(line, "Set at least one solver number."));
    };
    let nr_feature = count(&header, "features")?;
    let expected_rows = count(&header, "rows")?;
    let bias = scalar(&header, "bias")?;
    let regularize_bias = match single(&header, "regularize_bias")? {
        (_, "0") => false,
        (_, "1") => true,
        (line, _) => return Err(Error::input(line, "Set regularize_bias to 0 or 1.")),
    };
    let parameter = Parameter {
        solver_type,
        c: scalar(&header, "C")?,
        eps: scalar(&header, "eps")?,
        p: scalar(&header, "p")?,
        nu: scalar(&header, "nu")?,
        class_weights: weights(&header)?,
        regularize_bias,
        init_sol: None,
        w_recalc: false,
    };
    let mut labels = Vec::new();
    let mut rows = Vec::new();
    for (offset, line) in lines {
        if line.trim().is_empty() {
            continue;
        }
        let (label, row) = read_row(line, next_line(offset)?, nr_feature, bias)?;
        labels.push(label);
        rows.push(row);
    }
    if rows.len() != expected_rows {
        return Err(Error::input(
            0,
            "Set rows to the number of data rows in the file.",
        ));
    }
    Ok(Problem {
        solvers,
        parameter,
        nr_feature,
        bias,
        labels,
        rows,
    })
}

fn next_line(offset: usize) -> Result<usize, Error> {
    offset
        .checked_add(1)
        .ok_or_else(|| Error::input(0, "The line number overflowed. Use a shorter file."))
}

fn field<'a>(header: &'a Header<'a>, key: &str) -> Result<(usize, &'a [&'a str]), Error> {
    header
        .get(key)
        .map(|(line, values)| (*line, values.as_slice()))
        .ok_or_else(|| Error::input(0, "Add all required header fields from the case format."))
}

fn single<'a>(header: &'a Header<'a>, key: &str) -> Result<(usize, &'a str), Error> {
    let (line, values) = field(header, key)?;
    match values {
        [value] => Ok((line, value)),
        _ => Err(Error::input(line, "Set one value after this header key.")),
    }
}

fn number<T: FromStr>(text: &str, line: usize) -> Result<T, Error> {
    text.parse()
        .map_err(|_| Error::input(line, "Replace the value with a valid number."))
}

fn finite(text: &str, line: usize) -> Result<f64, Error> {
    let value: f64 = number(text, line)?;
    if !value.is_finite() {
        return Err(Error::input(line, "Use a finite number."));
    }
    Ok(value)
}

fn scalar(header: &Header<'_>, key: &str) -> Result<f64, Error> {
    let (line, value) = single(header, key)?;
    finite(value, line)
}

fn count(header: &Header<'_>, key: &str) -> Result<usize, Error> {
    let (line, value) = single(header, key)?;
    let value: usize = number(value, line)?;
    if value == 0 || value >= i32::MAX as usize {
        return Err(Error::input(line, "Use a count from 1 to 2147483646."));
    }
    Ok(value)
}

fn weights(header: &Header<'_>) -> Result<Vec<(i32, f64)>, Error> {
    let (line, values) = field(header, "weights")?;
    let Some((size, pairs)) = values.split_first() else {
        return Err(Error::input(
            line,
            "Set the weight count, then each label and weight.",
        ));
    };
    let size: usize = number(size, line)?;
    if pairs.len() % 2 != 0 || pairs.len() / 2 != size {
        return Err(Error::input(
            line,
            "Set the weight count to the number of label and weight pairs.",
        ));
    }
    pairs
        .as_chunks::<2>()
        .0
        .iter()
        .map(|pair| {
            let label = number(pair[0], line)?;
            let weight = finite(pair[1], line)?;
            if weight <= 0.0 {
                return Err(Error::input(
                    line,
                    "Set each class weight to a number greater than zero.",
                ));
            }
            Ok((label, weight))
        })
        .collect()
}

fn read_row(
    text: &str,
    line: usize,
    nr_feature: usize,
    bias: f64,
) -> Result<(f64, Vec<Feature>), Error> {
    let mut fields = text.split_whitespace();
    let label = finite(fields.next().unwrap_or(""), line)
        .map_err(|_| Error::input(line, "Set the row label to a finite number."))?;
    let mut row = Vec::new();
    let mut previous = 0;
    for field in fields {
        let Some((index, value)) = field.split_once(':') else {
            return Err(Error::input(line, "Write each feature as index:value."));
        };
        let index: usize = number(index, line)?;
        if index <= previous || index > nr_feature {
            return Err(Error::input(
                line,
                "Use feature indices from 1 to features in increasing order.",
            ));
        }
        row.push(Feature {
            index,
            value: finite(value, line)?,
        });
        previous = index;
    }
    if bias >= 0.0 {
        row.push(Feature {
            index: nr_feature.checked_add(1).ok_or_else(Error::overflow)?,
            value: bias,
        });
    }
    Ok((label, row))
}
