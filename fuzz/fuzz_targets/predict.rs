// SPDX-License-Identifier: BSD-3-Clause
#![no_main]

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    let Some(rows) = liblinear_fuzz::rows_from(data) else {
        return;
    };
    liblinear_fuzz::predict_rows(rows);
});
