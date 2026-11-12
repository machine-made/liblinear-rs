// SPDX-License-Identifier: BSD-3-Clause
//! Typed errors for a case file, a parameter or a model.
//!
//! ```
//! use liblinear::Error;
//! assert_eq!(Error::UnknownSolver(99).to_string(), "solver 99 does not exist.");
//! ```

use std::io;

use thiserror::Error;

/// An error from a case file, a parameter, or a model.
///
/// ```
/// use liblinear::Error;
/// assert_eq!(
///     Error::UnknownSolver(99).to_string(),
///     "solver 99 does not exist."
/// );
/// assert_eq!(
///     Error::Overflow.to_string(),
///     "A size calculation overflowed. Use fewer rows, classes, or features."
/// );
/// assert_eq!(
///     Error::Index { index: 3, length: 2 }.to_string(),
///     "Index 3 is outside length 2. Use an index below that length."
/// );
/// ```
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum Error {
    /// The file could not be read.
    ///
    /// ```
    /// let error = liblinear::Error::from(std::io::Error::new(
    ///     std::io::ErrorKind::NotFound,
    ///     "missing",
    /// ));
    /// assert!(error.to_string().contains("Cannot read the case file"));
    /// ```
    #[error("Cannot read the case file. Check the file and read access. ({0})")]
    Io(#[from] io::Error),
    /// This solver number does not exist.
    ///
    /// ```
    /// assert_eq!(
    ///     liblinear::Error::UnknownSolver(99).to_string(),
    ///     "solver 99 does not exist."
    /// );
    /// ```
    #[error("solver {0} does not exist.")]
    UnknownSolver(i32),
    /// The input must be changed. Line zero means no line is known.
    ///
    /// ```
    /// let error = liblinear::Error::InvalidInput {
    ///     line: 0,
    ///     message: "Set C to a number greater than zero.",
    /// };
    /// assert_eq!(error.to_string(), "Set C to a number greater than zero.");
    /// ```
    #[error("{}", invalid_text(*.line, .message))]
    InvalidInput {
        /// Line number. Zero means no line is known.
        ///
        /// ```
        /// let error = liblinear::Error::InvalidInput { line: 3, message: "Remove the repeated header key." };
        /// assert!(error.to_string().starts_with("Line 3:"));
        /// ```
        line: usize,
        /// Correction for the rejected input.
        ///
        /// ```
        /// let error = liblinear::Error::InvalidInput { line: 0, message: "Add at least one data row." };
        /// assert_eq!(error.to_string(), "Add at least one data row.");
        /// ```
        message: &'static str,
    },
    /// Two slices do not have the same length.
    ///
    /// ```
    /// let error = liblinear::Error::Length { left: 2, right: 1 };
    /// assert_eq!(
    ///     error.to_string(),
    ///     "Vector lengths 2 and 1 differ. Use two slices of equal length."
    /// );
    /// ```
    #[error("Vector lengths {left} and {right} differ. Use two slices of equal length.")]
    Length {
        /// Length of the first slice.
        ///
        /// ```
        /// let error = liblinear::Error::Length { left: 2, right: 1 };
        /// assert!(error.to_string().contains("2 and 1"));
        /// ```
        left: usize,
        /// Length of the second slice.
        ///
        /// ```
        /// let error = liblinear::Error::Length { left: 2, right: 1 };
        /// assert!(error.to_string().contains("equal length"));
        /// ```
        right: usize,
    },
    /// An index is outside its array.
    ///
    /// ```
    /// let error = liblinear::Error::Index { index: 3, length: 2 };
    /// assert_eq!(
    ///     error.to_string(),
    ///     "Index 3 is outside length 2. Use an index below that length."
    /// );
    /// ```
    #[error("Index {index} is outside length {length}. Use an index below that length.")]
    Index {
        /// The rejected index.
        ///
        /// ```
        /// let error = liblinear::Error::Index { index: 3, length: 2 };
        /// assert!(error.to_string().contains("Index 3"));
        /// ```
        index: usize,
        /// The array length.
        ///
        /// ```
        /// let error = liblinear::Error::Index { index: 3, length: 2 };
        /// assert!(error.to_string().contains("length 2"));
        /// ```
        length: usize,
    },
    /// A sum or product of sizes overflowed.
    ///
    /// ```
    /// assert_eq!(
    ///     liblinear::Error::Overflow.to_string(),
    ///     "A size calculation overflowed. Use fewer rows, classes, or features."
    /// );
    /// ```
    #[error("A size calculation overflowed. Use fewer rows, classes, or features.")]
    Overflow,
    /// A row, class, or feature count does not fit in 32 bits.
    ///
    /// ```
    /// let error = liblinear::Error::Count { count: usize::MAX };
    /// assert!(error.to_string().contains("does not fit in 32 bits"));
    /// ```
    #[error("Count {count} does not fit in 32 bits. Use fewer rows, classes, or features.")]
    Count {
        /// The rejected count.
        ///
        /// ```
        /// let error = liblinear::Error::Count { count: 9 };
        /// assert!(error.to_string().contains("Count 9"));
        /// ```
        count: usize,
    },
    /// A one-class model has no rho.
    ///
    /// ```
    /// assert_eq!(
    ///     liblinear::Error::MissingRho.to_string(),
    ///     "The one-class model has no rho. Train solver 21 before you score rows."
    /// );
    /// ```
    #[error("The one-class model has no rho. Train solver 21 before you score rows.")]
    MissingRho,
    /// A classification model has no labels.
    ///
    /// ```
    /// assert_eq!(
    ///     liblinear::Error::MissingLabels.to_string(),
    ///     "The classification model has no labels. Train a classifier before you predict labels."
    /// );
    /// ```
    #[error(
        "The classification model has no labels. Train a classifier before you predict labels."
    )]
    MissingLabels,
}

fn invalid_text(line: usize, message: &str) -> String {
    if line == 0 {
        message.to_owned()
    } else {
        format!("Line {line}: {message}")
    }
}

impl Error {
    pub(crate) fn input(line: usize, message: &'static str) -> Self {
        Self::InvalidInput { line, message }
    }

    pub(crate) fn overflow() -> Self {
        Self::Overflow
    }

    pub(crate) fn index(index: usize, length: usize) -> Self {
        Self::Index { index, length }
    }

    // Report the first index that lies past the array.
    pub(crate) fn range(start: usize, length: usize) -> Self {
        Self::index(if start < length { length } else { start }, length)
    }
}
