// SPDX-License-Identifier: BSD-3-Clause
#![no_main]

use libfuzzer_sys::fuzz_target;
use std::io::Write;

// The reader takes a path. One temp file carries these bytes to that call.
fuzz_target!(|data: &[u8]| {
    let mut path = std::env::temp_dir();
    path.push(format!("liblinear-read-problem-{}", std::process::id()));
    let mut file = match std::fs::File::create(&path) {
        Ok(file) => file,
        Err(_) => return,
    };
    if file.write_all(data).is_err() {
        drop(file);
        let _ = std::fs::remove_file(&path);
        return;
    }
    drop(file);
    let _ = liblinear::read_problem(&path);
    let _ = std::fs::remove_file(&path);
});
