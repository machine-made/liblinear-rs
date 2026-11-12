// SPDX-License-Identifier: BSD-3-Clause
use crate::rand::GlibcRand;
use serde_json::Value;
use std::io::{self, Write};

#[test]
fn matches_c_rand_for_two_seeds() {
    let table: Value = serde_json::from_str(include_str!("fixtures/rand_table.json")).unwrap();
    let seeds = table["seeds"].as_array().unwrap();
    assert_eq!(seeds.len(), 2);
    for (entry, expected_seed) in seeds.iter().zip([1_u32, 2]) {
        let seed = u32::try_from(entry["seed"].as_u64().unwrap()).unwrap();
        assert_eq!(seed, expected_seed);
        let mut stream = GlibcRand::new(seed).unwrap();
        let values = entry["values"].as_array().unwrap();
        assert_eq!(values.len(), 10_000);
        for (i, expected) in values.iter().enumerate() {
            assert_eq!(
                u64::from(stream.next_u31()),
                expected.as_u64().unwrap(),
                "seed {seed}, value {i}"
            );
        }
    }
    assert_eq!(GlibcRand::new(1).unwrap().next_u31(), 1_804_289_383);
    let mut zero = GlibcRand::new(0).unwrap();
    let mut one = GlibcRand::new(1).unwrap();
    for _ in 0..10_000 {
        assert_eq!(zero.next_u31(), one.next_u31());
    }
    writeln!(
        io::stdout().lock(),
        "rand: 10000 of 10000 equal for seed 1 and seed 2"
    )
    .unwrap();
}
