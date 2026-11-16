#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
"""Measure the unchanged C library and the Rust public API on the parity pairs.

Run under jobscope, with CARGO_BUILD_JOBS=2 and RAYON_NUM_THREADS=2:
  python3 -I /absolute/repo/scripts/speed_vs_c.py --self-test
  python3 -I /absolute/repo/scripts/speed_vs_c.py
  python3 -I /absolute/repo/scripts/speed_vs_c.py --rerun

LIBLINEAR_REAL_DATA_CACHE selects an outside-repo cache. --rerun retains the
first measurements and records all differences; it never averages two runs.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import platform
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REPORT = ROOT / "reference/perf/speed_vs_c.json"
REPEATS = 10

# Isolated Python omits the script folder from sys.path. Load this one trusted
# sibling explicitly, as the existing real-data script checks do.
sys.dont_write_bytecode = True
spec = importlib.util.spec_from_file_location("real_data_parity", ROOT / "scripts/real_data_parity.py")
assert spec is not None and spec.loader is not None
parity = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = parity
spec.loader.exec_module(parity)

# Include the unchanged reader rather than maintain another case-file parser.
# The translation unit lives only in the cache. The library sources and flags
# are those used by real_data_parity.build_c and reference/BUILD.md.
C_DRIVER = r'''
// SPDX-License-Identifier: BSD-3-Clause
#include <chrono>
#define main reference_driver_main
#include "REFERENCE_DRIVER"
#undef main
int main(int argc, char** argv) {
    try {
        require(argc == 2, "Use: speed_c CASE.txt");
        std::setlocale(LC_ALL, "C");
        set_print_string_function(quiet);
        Case c = read_case(argv[1]);
        require(c.solvers.size() == 1, "Select exactly one solver.");
        std::vector<feature_node*> x;
        for (auto& row : c.rows) x.push_back(row.data());
        problem prob{};
        prob.l = int(x.size());
        prob.n = c.features + (c.bias >= 0 ? 1 : 0);
        prob.y = c.labels.data();
        prob.x = x.data();
        prob.bias = c.bias;
        c.param.solver_type = c.solvers.front();
        c.param.nr_weight = int(c.class_weights.size());
        c.param.weight_label = c.weight_labels.data();
        c.param.weight = c.class_weights.data();
        const char* error = check_parameter(&prob, &c.param);
        if (error) throw std::runtime_error(error);
        std::srand(1);
        typedef std::chrono::steady_clock Clock;
        auto started = Clock::now();
        model* fitted = train(&prob, &c.param);
        double train_seconds = std::chrono::duration<double>(Clock::now() - started).count();
        require(fitted != nullptr, "The model is empty.");
        std::vector<double> predictions(prob.l);
        started = Clock::now();
        for (int i = 0; i < prob.l; ++i) predictions[i] = predict(fitted, prob.x[i]);
        double predict_seconds = std::chrono::duration<double>(Clock::now() - started).count();
        double checksum = 0;
        for (double value : predictions) checksum += value;
        std::printf("{\"train_seconds\":%.9f,\"predict_seconds\":%.9f,\"rows\":%d,\"checksum\":%.17g}\n",
                    train_seconds, predict_seconds, prob.l, checksum);
        free_and_destroy_model(&fitted);
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "%s\n", error.what());
        return 1;
    }
}
'''


def command(args: list[str]) -> str:
    return subprocess.run(args, check=True, capture_output=True, text=True).stdout.strip()


def prepare(cache: Path) -> tuple[dict, list[tuple[dict, Path]], Path]:
    manifest = parity.load_manifest()
    saved = json.loads((ROOT / "reference/parity/real_data.json").read_text())
    keys = lambda pairs: [(p["dataset"], p["solver"]) for p in pairs]
    if saved["status"] != "pass" or keys(saved["pairs"]) != keys(manifest["pairs"]):
        raise SystemExit("The manifest pairs must match the passing real-data parity report")
    if len(manifest["pairs"]) != 20 or len({p["solver"] for p in manifest["pairs"]}) != 12:
        raise SystemExit("Expected 20 pairs and 12 solvers")
    cache.mkdir(parents=True, exist_ok=True)
    datasets = {d["name"]: d for d in manifest["datasets"]}
    for dataset in datasets.values():
        parity.fetch_one(cache, dataset, False)
    cases = []
    for pair in manifest["pairs"]:
        dataset = datasets[pair["dataset"]]
        shape = parity.convert(cache, dataset, pair, manifest["parameters"])
        if shape != (dataset["rows"], dataset["features"]):
            raise SystemExit(f"Unexpected shape: {pair}")
        cases.append((pair, parity.case_path(cache, dataset, pair)))
    folder = cache / "_speed_build"
    folder.mkdir(exist_ok=True)
    source = folder / "speed_c.cpp"
    source.write_text(C_DRIVER.replace("REFERENCE_DRIVER", str(ROOT / "reference/driver/ref_driver.cpp")))
    binary = folder / "speed_c"
    subprocess.run(
        ["g++", "-std=c++11", "-O2", f"-I{parity.VENDOR}", str(source)]
        + [str(parity.VENDOR / s) for s in parity.SOURCES] + ["-o", str(binary)], check=True,
    )
    return manifest, cases, binary


def measure(binary: Path, case: Path, rss_path: Path) -> dict:
    # GNU time records this child's RSS. RUSAGE_CHILDREN instead retains the
    # largest earlier child, which would contaminate later measurements.
    output = subprocess.run(
        ["/usr/bin/time", "-f", "%M", "-o", str(rss_path), str(binary), str(case)],
        check=True, capture_output=True, text=True, timeout=parity.RUN_LIMIT_SECONDS,
    )
    result = json.loads(output.stdout)
    for field in ("train_seconds", "predict_seconds"):
        if not math.isfinite(result[field]) or result[field] <= 0:
            raise SystemExit(f"Invalid timing from {binary.name}: {result}")
    if not math.isfinite(result["checksum"]):
        raise SystemExit("The prediction checksum is not finite")
    result["peak_rss_kib"] = int(rss_path.read_text().strip())
    if result["peak_rss_kib"] <= 0:
        raise SystemExit("The process has no RSS measurement")
    return result


def interleave(left: Path, right: Path, case: Path, cache: Path) -> tuple[list, list]:
    rss_path = cache / "_speed_build/rss.txt"
    measure(left, case, rss_path)
    measure(right, case, rss_path)
    samples = ([], [])
    for _ in range(REPEATS):
        for index, binary in enumerate((left, right)):
            samples[index].append(measure(binary, case, rss_path))
    expected = samples[0][0]
    for side in samples:
        for sample in side:
            if sample["rows"] != expected["rows"] or sample["checksum"] != expected["checksum"]:
                raise SystemExit("The drivers disagree on prediction count or checksum")
    return samples


def summary(samples: list) -> dict:
    result = {"samples": samples}
    for field in ("train_seconds", "predict_seconds", "peak_rss_kib"):
        values = [s[field] for s in samples]
        result[field] = {"median": statistics.median(values), "min": min(values), "max": max(values)}
    return result


def rounded(value):
    if isinstance(value, float):
        return float(format(value, ".6g"))
    if isinstance(value, dict):
        return {k: rounded(v) for k, v in value.items()}
    if isinstance(value, list):
        return [rounded(v) for v in value]
    return value


def machine() -> dict:
    cpu = next(line.split(":", 1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines()
               if line.startswith("model name"))
    compiler = subprocess.run(["g++", "-v"], check=True, capture_output=True, text=True)
    # Compiler configure text can contain private build paths; the version
    # output is retained, with repository and home paths removed.
    verbose = (compiler.stdout + compiler.stderr).replace(str(ROOT), "<repo>")
    verbose = verbose.replace(str(Path.home()), "<home>")
    return {"cpu": cpu, "kernel": platform.release(),
            "architecture": platform.machine(), "rustc": command(["rustc", "-V"]),
            "g++": command(["g++", "--version"]).splitlines()[0], "g++_verbose": verbose.strip()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--rerun", action="store_true")
    args = parser.parse_args()
    if args.self_test and args.rerun:
        parser.error("Select --self-test or --rerun")
    for name in ("RAYON_NUM_THREADS", "CARGO_BUILD_JOBS"):
        if os.environ.get(name) != "2":
            raise SystemExit(f"Run under jobscope with {name}=2")
    os.nice(max(0, 19 - os.getpriority(os.PRIO_PROCESS, 0)))
    os.environ.update(LC_ALL="C", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    cache = parity.check_cache_outside(os.environ.get(
        "LIBLINEAR_REAL_DATA_CACHE", str(Path.home() / ".cache/liblinear-real-data")))
    previous = json.loads(REPORT.read_text()) if args.rerun else None
    if REPORT.exists() and not args.self_test and not args.rerun:
        raise SystemExit("A baseline exists. Use --rerun to preserve it")
    manifest, cases, c_binary = prepare(cache)
    if args.self_test:
        # The larger sparse case makes the train clock less sensitive to its
        # resolution. Both sides still execute the same fresh process.
        case = next(path for pair, path in cases if pair["dataset"] == "rcv1_train.binary" and pair["solver"] == 5)
        left, right = interleave(c_binary, c_binary, case, cache)
        ratio = statistics.median(s["train_seconds"] for s in right) / statistics.median(s["train_seconds"] for s in left)
        result = {"ratio": ratio, "range": [0.97, 1.03], "pass": 0.97 <= ratio <= 1.03,
                  "binary_sha256": parity.sha256_file(c_binary), "left": summary(left), "right": summary(right)}
        (cache / "_speed_build/self_test.json").write_text(json.dumps(rounded(result), indent=2, sort_keys=True) + "\n")
        print(f"self-test train ratio {ratio:.6g}; pass {result['pass']}", flush=True)
        if not result["pass"]:
            raise SystemExit(1)
        return
    self_test_path = cache / "_speed_build/self_test.json"
    if not self_test_path.exists() or not json.loads(self_test_path.read_text())["pass"]:
        raise SystemExit("Run --self-test successfully before the full benchmark")
    subprocess.run(["cargo", "build", "--manifest-path", str(ROOT / "Cargo.toml"), "--locked",
                    "--release", "--example", "speed_vs_c"], check=True)
    rust_binary = Path(os.environ.get("CARGO_TARGET_DIR", str(ROOT / "target"))) / "release/examples/speed_vs_c"
    provenance = {"machine": machine(), "commit": command(["git", "-C", str(ROOT), "rev-parse", "HEAD"]),
                  "manifest_sha256": parity.sha256_file(parity.MANIFEST),
                  "parity_report_sha256": parity.sha256_file(ROOT / "reference/parity/real_data.json"),
                  "rust_driver_sha256": parity.sha256_file(ROOT / "examples/speed_vs_c.rs"),
                  "script_sha256": parity.sha256_file(Path(__file__)),
                  "c_binary_sha256": parity.sha256_file(c_binary),
                  "rust_binary_sha256": parity.sha256_file(rust_binary),
                  "datasets": [{"name": d["name"], "download_sha256": d["download_sha256"],
                                "text_sha256": d["text_sha256"]} for d in manifest["datasets"]],
                  "cases": [{"dataset": p["dataset"], "solver": p["solver"], "sha256": parity.sha256_file(c)} for p, c in cases]}
    if previous is not None:
        # A commit can add this report or change prose without changing the
        # measured programs. The hashes still require the same experiment.
        comparable = lambda data: {key: value for key, value in data.items() if key != "commit"}
        if comparable(previous["provenance"]) != comparable(provenance):
            raise SystemExit("Rerun provenance differs from the baseline; preserve both runs separately")
    pairs = []
    for pair, case in cases:
        c_samples, rust_samples = interleave(c_binary, rust_binary, case, cache)
        c, rust = summary(c_samples), summary(rust_samples)
        ratios = {field: rust[field]["median"] / c[field]["median"]
                  for field in ("train_seconds", "predict_seconds", "peak_rss_kib")}
        pairs.append({**pair, "c": c, "rust": rust, "ratios": ratios,
                      "follow_up": ratios["train_seconds"] > 2 or ratios["peak_rss_kib"] > 1.5})
        print(f"{pair['dataset']} s{pair['solver']}: train {ratios['train_seconds']:.6g} "
              f"predict {ratios['predict_seconds']:.6g} RSS {ratios['peak_rss_kib']:.6g}", flush=True)
    run = {"utc": datetime.now(timezone.utc).isoformat(), "commit": provenance["commit"], "pairs": pairs}
    if previous is None:
        report = {"schema_version": 1, "provenance": provenance,
                  "method": {"repeats": REPEATS, "warmups_per_binary_per_pair": 1,
                             "order": "C then Rust, interleaved", "nice": os.getpriority(os.PRIO_PROCESS, 0),
                             "threads_per_process": 1, "c_flags": ["-std=c++11", "-O2"], "rust_profile": "release",
                             "clock": "in-process monotonic wall clock, calls only",
                             "spread": "min and max of 10 samples", "rss": "median process peak RSS, GNU time %M, KiB",
                             "train_note": "Rust train also validates inputs and scores training rows; C train does not",
                             "predict_note": "Rust predict includes result allocation; C loops over predict with a preallocated result vector",
                             "rss_note": "Whole process, including case reader and model; not solver workspace alone"},
                  "self_test": json.loads(self_test_path.read_text()), "baseline": run}
    else:
        changes = []
        for before, after in zip(previous["baseline"]["pairs"], pairs, strict=True):
            for field in ("train_seconds", "predict_seconds"):
                change = abs(after["ratios"][field] / before["ratios"][field] - 1)
                changes.append({"dataset": after["dataset"], "solver": after["solver"],
                                "metric": field, "relative_change": change, "within_10_percent": change <= 0.1})
        report = previous
        report["rerun"] = run
        report["repeatability"] = {"changes": changes, "all_within_10_percent": all(c["within_10_percent"] for c in changes),
                                   "max_relative_change": max(c["relative_change"] for c in changes),
                                   "note": "WSL2 scheduler and shared-machine load can affect short calls; no runs are averaged"}
    REPORT.parent.mkdir(exist_ok=True)
    REPORT.write_text(json.dumps(rounded(report), indent=2, sort_keys=True, allow_nan=False) + "\n")
    if previous is not None:
        print(json.dumps(report["repeatability"], sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
