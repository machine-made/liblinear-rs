// SPDX-License-Identifier: BSD-3-Clause
//! Measure the public train and predict calls without case-file I/O.

use liblinear::{read_problem, train};
use std::{env, hint::black_box, time::Instant};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut arguments = env::args_os().skip(1);
    let path = arguments.next().ok_or("Use: speed_vs_c CASE.txt")?;
    if arguments.next().is_some() {
        return Err("Use: speed_vs_c CASE.txt".into());
    }
    let problem = read_problem(path)?;
    if problem.solvers.len() != 1 {
        return Err("The case must select exactly one solver".into());
    }
    let mut parameter = problem.parameter.clone();
    parameter.solver_type = problem.solvers.first().copied().ok_or("No solver")?;
    let started = Instant::now();
    let model = train(black_box(&problem), black_box(&parameter))?;
    let train_seconds = started.elapsed().as_secs_f64();
    let started = Instant::now();
    let predictions = model.predict(black_box(&problem))?;
    let predict_seconds = started.elapsed().as_secs_f64();
    black_box(&predictions);
    // A checksum keeps the result observable without timing output formatting.
    let checksum: f64 = predictions.iter().sum();
    println!(
        "{{\"train_seconds\":{train_seconds:.9},\"predict_seconds\":{predict_seconds:.9},\"rows\":{},\"checksum\":{checksum:.17}}}",
        predictions.len()
    );
    Ok(())
}
