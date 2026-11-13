// SPDX-License-Identifier: BSD-3-Clause
//! Trains every pair of a real dataset and a solver and compares the model with
//! the one saved from the C `train` program built at -O2. The datasets are not
//! in the repository. `scripts/real_data_parity.py` downloads them into a cache
//! folder and saves the C results there. The comparison is by bits: 0 ULP.

#![expect(
    clippy::indexing_slicing,
    clippy::unwrap_used,
    reason = "The test counts pairs and values and reads files named by the manifest"
)]

use liblinear::{SolverType, read_problem, train};
use serde_json::{Value, json};
use std::{
    collections::BTreeSet,
    env, fs,
    path::{Path, PathBuf},
    time::Instant,
};

type RunError = Box<dyn std::error::Error + Send + Sync>;

const CACHE_VARIABLE: &str = "LIBLINEAR_REAL_DATA_CACHE";
const TIMING_NOTE: &str = "Both times include reading the data file. C seconds cover the whole \
    train process, with the model write. Rust seconds cover read_problem and train, and train \
    also scores the training rows. Rust runs in the unoptimized test profile. Timing is not a \
    speed claim.";

fn cache() -> Result<PathBuf, RunError> {
    if let Some(path) = env::var_os(CACHE_VARIABLE) {
        return Ok(PathBuf::from(path));
    }
    let home = env::var_os("HOME").ok_or("Set HOME or LIBLINEAR_REAL_DATA_CACHE")?;
    Ok(PathBuf::from(home).join(".cache/liblinear-real-data"))
}

fn load(path: &Path) -> Result<Value, RunError> {
    let bytes = fs::read(path).map_err(|error| {
        format!(
            "Cannot read {}: {error}. Run scripts/real_data_parity.py with the modes fetch and c.",
            path.display()
        )
    })?;
    Ok(serde_json::from_slice(&bytes)?)
}

fn ordered_bits(value: f64) -> u64 {
    let bits = value.to_bits();
    if bits >> 63 == 0 {
        bits | (1 << 63)
    } else {
        !bits
    }
}

/// Distance in representable doubles. 0 means equal bits.
fn ulp(left: f64, right: f64) -> u64 {
    ordered_bits(left).abs_diff(ordered_bits(right))
}

fn numbers(value: &Value) -> Result<Vec<f64>, RunError> {
    let mut out = Vec::new();
    for item in value
        .as_array()
        .ok_or("The saved result has no number list")?
    {
        // The strings are the %.17g text of the C library. Rust parses them exactly.
        out.push(
            item.as_str()
                .ok_or("A saved number is not text")?
                .parse::<f64>()?,
        );
    }
    Ok(out)
}

struct Outcome {
    worst_ulp: u64,
    first_difference: Option<Value>,
}

impl Outcome {
    fn field(&mut self, name: &str, saved: &[f64], rust: &[f64]) {
        if saved.len() != rust.len() {
            self.differ(
                json!({"field": name, "what": "length", "c": saved.len(), "rust": rust.len()}),
            );
            return;
        }
        for (index, (&c, &r)) in saved.iter().zip(rust).enumerate() {
            let distance = ulp(c, r);
            self.worst_ulp = self.worst_ulp.max(distance);
            if distance != 0 {
                self.differ(json!({"field": name, "index": index, "c": c, "rust": r,
                    "c_bits": format!("{:016x}", c.to_bits()),
                    "rust_bits": format!("{:016x}", r.to_bits())}));
            }
        }
    }

    fn differ(&mut self, difference: Value) {
        if self.first_difference.is_none() {
            self.first_difference = Some(difference);
        }
    }
}

fn run_pair(cache: &Path, dataset: &Value, pair: &Value) -> Result<Value, RunError> {
    let name = dataset["name"].as_str().ok_or("A dataset has no name")?;
    let solver_number = i32::try_from(pair["solver"].as_i64().ok_or("A pair has no solver")?)?;
    let folder = cache.join(name);
    let saved = load(&folder.join(format!("c_solver{solver_number}.json")))?;
    let case = folder.join(format!("solver{solver_number}.case.txt"));

    let started = Instant::now();
    let problem = read_problem(&case)?;
    let mut parameter = problem.parameter.clone();
    parameter.solver_type = SolverType::try_from(solver_number)?;
    let model = train(&problem, &parameter)?;
    let rust_seconds = started.elapsed().as_secs_f64();

    let mut outcome = Outcome {
        worst_ulp: 0,
        first_difference: None,
    };
    let rows = problem.labels.len();
    let features = problem.nr_feature;
    if rows != saved["rows"]
        || features != saved["features"]
        || rows != dataset["rows"]
        || features != dataset["features"]
    {
        outcome.differ(
            json!({"field": "shape", "c_rows": saved["rows"], "c_features": saved["features"],
            "rust_rows": rows, "rust_features": features}),
        );
    }
    if model.nr_class != saved["nr_class"] {
        outcome
            .differ(json!({"field": "nr_class", "c": saved["nr_class"], "rust": model.nr_class}));
    }
    let saved_bias = saved["bias"]
        .as_str()
        .ok_or("The saved bias is not text")?
        .parse::<f64>()?;
    if model.nr_feature != saved["nr_feature"] || ulp(model.bias, saved_bias) != 0 {
        outcome.differ(
            json!({"field": "nr_feature or bias", "c_nr_feature": saved["nr_feature"],
            "c_bias": saved_bias, "rust_nr_feature": model.nr_feature, "rust_bias": model.bias}),
        );
    }
    let saved_labels: Option<Vec<i32>> = match saved["labels"].as_array() {
        None => None,
        Some(list) => Some(
            list.iter()
                .map(|label| {
                    Ok(label
                        .as_str()
                        .ok_or("A label is not text")?
                        .parse::<i32>()?)
                })
                .collect::<Result<_, RunError>>()?,
        ),
    };
    if saved_labels != model.labels {
        outcome.differ(json!({"field": "labels", "c": saved_labels, "rust": model.labels}));
    }
    outcome.field("weights", &numbers(&saved["weights"])?, &model.weights);
    let saved_rho = match saved["rho"].as_str() {
        None => Vec::new(),
        Some(text) => vec![text.parse::<f64>()?],
    };
    outcome.field("rho", &saved_rho, model.rho.as_slice());
    outcome.field(
        "predictions",
        &numbers(&saved["predictions"])?,
        &model.predictions,
    );

    let rounded = |seconds: f64| (seconds * 1000.0).round() / 1000.0;
    Ok(json!({
        "dataset": name,
        "solver": solver_number,
        "rows": rows,
        "features": features,
        "c_seconds": rounded(saved["c_seconds"].as_f64().ok_or("The saved result has no time")?),
        "rust_seconds": rounded(rust_seconds),
        "worst_ulp": outcome.worst_ulp,
        "equal": outcome.first_difference.is_none(),
        "weights": model.weights.len(),
        "weights_sha256": saved["weights_sha256"],
        "first_difference": outcome.first_difference,
    }))
}

#[test]
#[ignore = "needs the cached datasets and C results; see docs/PARITY.md"]
fn every_pair_matches_c() -> Result<(), RunError> {
    let repo = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let manifest = load(&repo.join("reference/real_data_manifest.json"))?;
    let cache = cache()?;
    let datasets = manifest["datasets"]
        .as_array()
        .ok_or("The manifest has no datasets")?;
    let pairs = manifest["pairs"]
        .as_array()
        .ok_or("The manifest has no pairs")?;
    let mut lines = Vec::new();
    let mut worst = 0_u64;
    let mut differing = Vec::new();
    let mut used = BTreeSet::new();
    for pair in pairs {
        let dataset = datasets
            .iter()
            .find(|dataset| dataset["name"] == pair["dataset"])
            .ok_or("A pair names an unknown dataset")?;
        let result = run_pair(&cache, dataset, pair)?;
        println!(
            "{} solver {}: rows {} features {} C {} s Rust {} s worst {} ULP equal {}",
            result["dataset"].as_str().unwrap(),
            result["solver"],
            result["rows"],
            result["features"],
            result["c_seconds"],
            result["rust_seconds"],
            result["worst_ulp"],
            result["equal"]
        );
        used.insert(result["dataset"].as_str().unwrap().to_owned());
        worst = worst.max(result["worst_ulp"].as_u64().unwrap());
        if result["equal"] != true {
            differing.push(format!("{} solver {}", result["dataset"], result["solver"]));
        }
        lines.push(format!("    {}", serde_json::to_string(&result)?));
    }
    // A run that covers fewer pairs than the manifest lists must fail, not pass.
    assert!(!pairs.is_empty(), "the manifest lists no pairs");
    assert_eq!(
        lines.len(),
        pairs.len(),
        "the run compared fewer pairs than the manifest lists"
    );
    for dataset in datasets {
        let name = dataset["name"].as_str().ok_or("A dataset has no name")?;
        assert!(used.contains(name), "no pair used the dataset {name}");
    }
    let report = format!(
        "{{\n  \"status\": \"{}\",\n  \"tolerance\": \"0 ULP: equal bits\",\n  \"manifest\": \
         \"reference/real_data_manifest.json\",\n  \"pair_count\": {},\n  \"equal_count\": {},\n  \
         \"worst_ulp\": {},\n  \"timing_note\": {},\n  \"pairs\": [\n{}\n  ]\n}}\n",
        if differing.is_empty() {
            "pass"
        } else {
            "differences"
        },
        lines.len(),
        lines.len().saturating_sub(differing.len()),
        worst,
        serde_json::to_string(TIMING_NOTE)?,
        lines.join(",\n")
    );
    fs::write(repo.join("reference/parity/real_data.json"), report)?;
    assert!(
        differing.is_empty(),
        "pairs that differ from C: {differing:?}"
    );
    Ok(())
}
