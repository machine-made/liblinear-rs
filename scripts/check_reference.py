# SPDX-License-Identifier: BSD-3-Clause
"""Check the frozen inputs, results, source hashes and build spread."""

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOLVERS = {0, 1, 2, 3, 4, 5, 6, 7, 11, 12, 13, 21}
ENV = dict(os.environ, RAYON_NUM_THREADS="2", CARGO_BUILD_JOBS="2", LC_ALL="C")
SOURCES = [
    "reference/driver/ref_driver.cpp",
    "vendor/liblinear/linear.cpp",
    "vendor/liblinear/newton.cpp",
] + [f"vendor/liblinear/blas/{name}.c" for name in ("daxpy", "ddot", "dnrm2", "dscal")]


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path):
    def invalid(_):
        raise ValueError("The JSON value is not finite.")

    return json.loads(path.read_text(), parse_constant=invalid)


def saved_files(reference):
    return sorted(
        [
            reference / "BUILD.md",
            reference / "spread.json",
            reference / "driver/ref_driver.cpp",
        ]
        + list((reference / "corpus").glob("*.txt"))
        + list((reference / "golden/O0").glob("*.json"))
        + list((reference / "golden/O2").glob("*.json"))
    )


def write_manifest(reference):
    (reference / "MANIFEST.sha256").write_text(
        "".join(
            f"{digest(path)}  {path.relative_to(reference).as_posix()}\n"
            for path in saved_files(reference)
        )
    )


def check_manifest(reference):
    manifest = reference / "MANIFEST.sha256"
    require(manifest.is_file(), "The manifest is missing.")
    entries = {}
    for line in manifest.read_text().splitlines():
        match = re.fullmatch(r"([0-9a-f]{64})  (.+)", line)
        require(match is not None, "The manifest line is not valid.")
        sha, name = match.groups()
        path = Path(name)
        require(
            not path.is_absolute() and ".." not in path.parts and name not in entries,
            "The manifest path is not valid.",
        )
        entries[name] = sha
    actual = {p.relative_to(reference).as_posix() for p in saved_files(reference)}
    missing = sum(not (reference / name).is_file() for name in entries)
    mismatches = sum(
        (reference / name).is_file() and digest(reference / name) != sha
        for name, sha in entries.items()
    )
    require(
        missing == 0 and mismatches == 0,
        f"{missing} missing, {mismatches} hash mismatches",
    )
    require(actual == set(entries), "The manifest file list differs.")


def check_source():
    vendor = ROOT / "vendor/liblinear"
    provenance = (vendor / "PROVENANCE.md").read_text()
    require(
        "0ce1759d64082ab79905cf89ccc7b4443f43d117" in provenance
        and "491c9f1188b97ba70847c70a68be363d186ddf9d" in provenance,
        "The source version differs.",
    )
    entries = re.findall(r"^([0-9a-f]{64})  (.+)$", provenance, re.MULTILINE)
    require(len(entries) == 17, "The source hash list is not complete.")
    for sha, name in entries:
        require(
            (vendor / name).is_file() and digest(vendor / name) == sha,
            f"The source hash differs: {name}",
        )
    require(
        (ROOT / "THIRD-PARTY-NOTICES").read_bytes()
        == (vendor / "COPYRIGHT").read_bytes(),
        "The copyright copy differs.",
    )
    require(not (ROOT / "src").exists(), "The src folder must not exist yet.")


def case_settings(path):
    header, body = path.read_text().split("\ndata\n", 1)
    settings = dict(line.split(" ", 1) for line in header.splitlines())
    settings["solvers"] = list(map(int, settings["solvers"].split()))
    settings["rows"] = int(settings["rows"])
    settings["features"] = int(settings["features"])
    require(len(body.splitlines()) == settings["rows"], "The row count differs.")
    return settings


def check_model(data, settings):
    require(
        set(data) == {"runs"} and isinstance(data["runs"], list),
        "The result format differs.",
    )
    require(
        [r["solver"] for r in data["runs"]] == settings["solvers"],
        "The result solver list differs.",
    )
    for run in data["runs"]:
        require(
            set(run)
            == {
                "solver",
                "C",
                "eps",
                "bias",
                "nr_class",
                "nr_feature",
                "rho",
                "labels",
                "weights",
                "decision_values",
                "predictions",
            },
            "The result fields differ.",
        )
        require(
            all(run[key] == float(settings[key]) for key in ("C", "eps", "bias")),
            "The result options differ.",
        )
        require(
            type(run["nr_class"]) is int
            and run["nr_class"] > 0
            and run["nr_feature"] == settings["features"],
            "The model size differs.",
        )
        columns = 1 if run["nr_class"] == 2 and run["solver"] != 4 else run["nr_class"]
        require(
            len(run["weights"])
            == (settings["features"] + (run["bias"] >= 0)) * columns,
            "The weight count differs.",
        )
        require(
            len(run["predictions"]) == settings["rows"]
            and len(run["decision_values"]) == settings["rows"]
            and all(len(values) == columns for values in run["decision_values"]),
            "The prediction size differs.",
        )
        if run["solver"] in (11, 12, 13, 21):
            require(run["labels"] is None, "This solver must have no class labels.")
        else:
            require(
                isinstance(run["labels"], list)
                and len(run["labels"]) == run["nr_class"],
                "The class label count differs.",
            )
        require(
            (run["rho"] is not None) == (run["solver"] == 21), "The rho field differs."
        )


def measure_spread(left, right):
    """Measure each numeric output. Group array entries under the same path."""
    outputs = {}

    def walk(a, b, path):
        numeric = type(a) in (int, float) and type(b) in (int, float)
        require(numeric or type(a) is type(b), f"The output type differs: {path}")
        if isinstance(a, dict):
            require(a.keys() == b.keys(), f"The output keys differ: {path}")
            for key in a:
                walk(a[key], b[key], f"{path}.{key}" if path else key)
        elif isinstance(a, list):
            require(len(a) == len(b), f"The output length differs: {path}")
            for x, y in zip(a, b):
                walk(x, y, path + "[]")
        elif numeric:
            require(math.isfinite(a) and math.isfinite(b), "The output is not finite.")
            absolute = abs(b - a)
            scale = max(abs(a), abs(b))
            relative = absolute / scale if scale else 0.0
            stats = outputs.setdefault(
                path, {"count": 0, "max_absolute": 0.0, "max_relative": 0.0}
            )
            stats["count"] += 1
            stats["max_absolute"] = max(stats["max_absolute"], absolute)
            stats["max_relative"] = max(stats["max_relative"], relative)
        else:
            require(a == b, f"The output value differs: {path}")

    walk(left, right, "")
    return outputs


def spread_report(reference):
    cases = {}
    totals = {}
    for path in sorted((reference / "corpus").glob("*.txt")):
        name = path.stem + ".json"
        stats = measure_spread(
            read_json(reference / "golden/O0" / name),
            read_json(reference / "golden/O2" / name),
        )
        cases[path.stem] = stats
        for field, item in stats.items():
            total = totals.setdefault(
                field, {"count": 0, "max_absolute": 0.0, "max_relative": 0.0}
            )
            total["count"] += item["count"]
            for key in ("max_absolute", "max_relative"):
                total[key] = max(total[key], item[key])
    return {
        "builds": ["O0", "O2"],
        "cases": len(cases),
        "relative_difference": "abs(O2 - O0) / max(abs(O0), abs(O2)); zero if both are zero",
        "outputs": totals,
        "per_case": cases,
    }


def check_analytic(reference, announce=False):
    expected = 4000 / 4001
    for level in ("O0", "O2"):
        path = reference / f"golden/{level}/analytic_l2svc.json"
        for run in read_json(path)["runs"]:
            actual = run["weights"][0]
            difference = abs(actual - expected)
            if difference > 1e-3:
                raise ValueError(
                    f"Analytic solver {run['solver']}: w = {actual:.17g}; "
                    "stop the freeze. Call: ref_driver "
                    "reference/corpus/analytic_l2svc.txt OUTPUT.json"
                )
            require(difference < 1e-4, "The analytic weight is outside 1e-4.")
            if announce and level == "O2":
                print(
                    f"solver {run['solver']}: analytic w = {actual:.5f} "
                    f"(expected {expected:.5f}, difference below 1e-4)"
                )


def validate(reference, regenerate=True):
    check_manifest(reference)
    check_source()
    cases = sorted((reference / "corpus").glob("*.txt"))
    coverage = Counter()
    names = {path.stem + ".json" for path in cases}
    for level in ("O0", "O2"):
        require(
            {p.name for p in (reference / f"golden/{level}").glob("*.json")} == names,
            f"The {level} result file list differs.",
        )
        for path in cases:
            settings = case_settings(path)
            data = read_json(reference / f"golden/{level}" / (path.stem + ".json"))
            check_model(data, settings)
            coverage.update(r["solver"] for r in data["runs"])
    require(set(coverage) == SOLVERS, "A solver is missing from the reference.")
    require(
        read_json(reference / "spread.json") == spread_report(reference),
        "The saved build spread differs.",
    )
    check_analytic(reference)
    if regenerate:
        with tempfile.TemporaryDirectory(prefix="liblinear-corpus-") as folder:
            output = Path(folder)
            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts/gen_corpus.py"),
                    "--output",
                    str(output),
                ],
                check=True,
                capture_output=True,
                env=ENV,
            )
            expected = {p.name for p in output.glob("*.txt")}
            require(
                expected == {p.name for p in cases}, "The corpus file list differs."
            )
            require(
                all(p.read_bytes() == (output / p.name).read_bytes() for p in cases),
                "The corpus differs from the fixed seed.",
            )


def build_o2(output):
    version = subprocess.check_output(
        ["g++", "-dumpfullversion"], text=True, env=ENV
    ).strip()
    require(version == "13.3.0", "Use g++ 13.3.0 to check the reference.")
    subprocess.run(
        [
            "g++",
            "-std=c++11",
            "-O2",
            f"-I{ROOT / 'vendor/liblinear'}",
            *(str(ROOT / name) for name in SOURCES),
            "-o",
            str(output),
        ],
        check=True,
        env=ENV,
    )


def run_case(driver, case, output):
    subprocess.run(
        [str(driver), str(case), str(output)], check=True, env=ENV, timeout=120
    )


def rerun(reference, driver, folder):
    for case in sorted((reference / "corpus").glob("*.txt")):
        first, second = folder / "first.json", folder / "second.json"
        run_case(driver, case, first)
        run_case(driver, case, second)
        require(
            first.read_bytes() == second.read_bytes(), f"The rerun differs: {case.name}"
        )
        golden = reference / "golden/O2" / (case.stem + ".json")
        require(
            first.read_bytes() == golden.read_bytes(),
            f"The O2 result differs: {case.name}",
        )


def record(reference, drivers):
    for level, driver in zip(("O0", "O2"), drivers):
        require(
            driver is not None and driver.is_file(), "A reference driver is missing."
        )
        output = reference / f"golden/{level}"
        output.mkdir(parents=True, exist_ok=True)
        analytic = reference / "corpus/analytic_l2svc.txt"
        run_case(driver, analytic, output / "analytic_l2svc.json")
        for run in read_json(output / "analytic_l2svc.json")["runs"]:
            actual = run["weights"][0]
            require(
                abs(actual - 4000 / 4001) < 1e-4,
                f"Analytic solver {run['solver']}: w = {actual:.17g}; "
                f"call: {driver} {analytic} {output / 'analytic_l2svc.json'}",
            )
        for case in sorted((reference / "corpus").glob("*.txt")):
            if case != analytic:
                run_case(driver, case, output / (case.stem + ".json"))
        print(
            f"{level}: wrote {len(list(output.glob('*.json')))} result files.",
            flush=True,
        )
    (reference / "spread.json").write_text(
        json.dumps(spread_report(reference), indent=2, allow_nan=False) + "\n"
    )
    write_manifest(reference)
    validate(reference)


def self_test(reference):
    validate(reference)
    with tempfile.TemporaryDirectory(prefix="liblinear-self-test-") as folder:
        copied = Path(folder) / "reference"
        shutil.copytree(reference, copied)
        golden = copied / "golden/O2/s00_dense.json"
        original = golden.read_bytes()
        altered = bytearray(original)
        altered[0] ^= 1
        golden.write_bytes(altered)
        result = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--reference", str(copied)],
            check=False,
            capture_output=True,
            text=True,
            env=ENV,
        )
        require(
            result.returncode == 1 and "1 hash mismatches" in result.stderr,
            "The changed-byte test did not fail as required.",
        )
        print("self-test: changed golden byte: exit 1 (1 hash mismatches)")
        golden.write_bytes(original)
        for case in (copied / "corpus").glob("s21_*.txt"):
            for level in ("O0", "O2"):
                (copied / f"golden/{level}" / (case.stem + ".json")).unlink()
            case.unlink()
        (copied / "spread.json").write_text(
            json.dumps(spread_report(copied), indent=2) + "\n"
        )
        write_manifest(copied)
        result = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--reference", str(copied)],
            check=False,
            capture_output=True,
            text=True,
            env=ENV,
        )
        require(
            result.returncode == 1 and "solver is missing" in result.stderr,
            "The missing-solver test did not fail as required.",
        )
        print("self-test: missing solver 21: exit 1 (manifest and spread rebuilt)")
    require(
        measure_spread({"x": [0.0, 2.0]}, {"x": [1.0, 3.0]})
        == {"x[]": {"count": 2, "max_absolute": 1.0, "max_relative": 1.0}},
        "The spread calculation test failed.",
    )
    print("self-test: spread calculation: passed")
    adjacent = math.nextafter(1.0, math.inf)
    require(
        measure_spread({"weight": 1}, {"weight": adjacent})
        == {
            "weight": {
                "count": 1,
                "max_absolute": adjacent - 1,
                "max_relative": (adjacent - 1) / adjacent,
            }
        },
        "The mixed numeric token test failed.",
    )
    print("self-test: mixed numeric tokens: passed")
    print("self-test: passed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, default=ROOT / "reference")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--record", action="store_true")
    parser.add_argument("--driver-o0", type=Path)
    parser.add_argument("--driver-o2", type=Path)
    args = parser.parse_args()
    reference = args.reference.resolve()
    if args.self_test:
        self_test(reference)
    elif args.record:
        record(
            reference,
            [p.resolve() if p else None for p in (args.driver_o0, args.driver_o2)],
        )
    else:
        validate(reference)
        with tempfile.TemporaryDirectory(prefix="liblinear-check-") as folder:
            folder = Path(folder)
            driver = (
                args.driver_o2.resolve() if args.driver_o2 else folder / "ref_driver_O2"
            )
            if not args.driver_o2:
                build_o2(driver)
            rerun(reference, driver, folder)
        print("12 solvers, 0 missing, 0 hash mismatches, 2 builds, rerun identical")
        check_analytic(reference, announce=True)


if __name__ == "__main__":
    try:
        main()
    except (
        ValueError,
        OSError,
        KeyError,
        TypeError,
        subprocess.SubprocessError,
    ) as error:
        print(f"Reference check failed: {error}", file=sys.stderr)
        sys.exit(1)
