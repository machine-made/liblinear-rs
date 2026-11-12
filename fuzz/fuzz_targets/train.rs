// SPDX-License-Identifier: BSD-3-Clause
#![no_main]

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    let Some((problem, parameter)) = liblinear_fuzz::problem_from(data) else {
        return;
    };
    for solver in liblinear_fuzz::SOLVERS {
        let mut parameter = parameter.clone();
        parameter.solver_type = solver;
        let mut problem = problem.clone();
        problem.parameter.solver_type = solver;
        problem.solvers = vec![solver];
        let _ = liblinear::train(&problem, &parameter);
    }
});
