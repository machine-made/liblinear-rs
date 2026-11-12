# SPDX-License-Identifier: BSD-3-Clause
"""Build the unchanged C++ source and save six SVR results and branch counts."""
import gzip
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "tests/fixtures"
BUILD = ROOT / "target/svr-reference"
BUILD.mkdir(parents=True, exist_ok=True)
ENV = dict(os.environ, RAYON_NUM_THREADS="2", CARGO_BUILD_JOBS="2", LC_ALL="C")


def run(args):
    return subprocess.run([str(x) for x in args], cwd=BUILD, env=ENV,
                          check=True, capture_output=True).stdout


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def coverage():
    run(["gcov", "-j", "-b", BUILD / "reference_O0-linear.gcno"])
    data = json.loads(gzip.decompress((BUILD / "reference_O0-linear.gcov.json.gz").read_bytes()))
    lines = next(f["lines"] for f in data["files"] if f["file"].endswith("/linear.cpp"))
    # Entries into zero, upper and lower bound shrinking, reset, and limit exits.
    return {str(v["line_number"]): v["count"] for v in lines
            if v["line_number"] in [1200, 1212, 1224, 1269, 2742, 2751]}


rng = random.Random(1113)
problems = []
specs = [("equal_targets", 1, 1e-8, 0.1),
         ("outlier", 1, 1e-8, 0.2),
         ("p_zero", 1, 1e-8, 0),
         ("p_large", 1, 1e-8, 100),
         ("duplicate_C_small", 1e-3, 1e-8, 0.1),
         ("zero_column_C_large", 1e3, 1e-30, 0)]
for name, cost, eps, p in specs:
    rows = []
    for i in range(12):
        values = [rng.uniform(-1, 1) for _ in range(5)]
        if name == "zero_column_C_large":
            values[2] = 0
        target = 2.5 if name == "equal_targets" else rng.uniform(-3, 3)
        if name == "outlier":
            target = 7.25 if i == 11 else 0
        rows.append((target, values))
    if name == "duplicate_C_small":
        rows[1] = rows[0]
    text = (f"solvers 11 12 13\nC {cost:.17g}\neps {eps:.17g}\nbias 1\np {p:.17g}\nnu 0.5\n"
            f"regularize_bias 1\nweights 0\nfeatures 5\nrows {len(rows)}\ndata\n")
    for label, values in rows:
        text += f"{label:.17g}" + "".join(f" {j}:{v:.17g}" for j, v in enumerate(values, 1) if v != 0) + "\n"
    path = FIXTURES / f"svr_{name}.txt"
    path.write_text(text)
    problems.append((name, path))

sources = [FIXTURES / "svr_reference.cpp", ROOT / "vendor/liblinear/linear.cpp",
           ROOT / "vendor/liblinear/newton.cpp"]
sources += [ROOT / f"vendor/liblinear/blas/{name}.c" for name in ["daxpy", "ddot", "dnrm2", "dscal"]]
commands = []
for flags, level in [(["-O0", "--coverage"], "O0"), (["-O2"], "O2")]:
    command = ["g++", "-std=c++11", *flags, "-I" + str(ROOT / "vendor/liblinear"),
               *sources, "-o", BUILD / f"reference_{level}"]
    run(command)
    commands.append("RAYON_NUM_THREADS=2 CARGO_BUILD_JOBS=2 " + " ".join(str(x).replace(str(ROOT), "{repo}") for x in command))


def reset_counts():
    for path in BUILD.glob("*.gcda"):
        path.unlink()


corpus_coverage = {}
for solver in [11, 12, 13]:
    reset_counts()
    for path in sorted((ROOT / "reference/corpus").glob("*.txt")):
        header = path.read_text().splitlines()[0].split()[1:]
        if str(solver) in header:
            run([BUILD / "reference_O0", path, BUILD / "corpus.json", solver])
    corpus_coverage[str(solver)] = coverage()
entries = []
extra_coverage = {}
raw_hashes = {}
for name, path in problems:
    reset_counts()
    for level in ["O0", "O2"]:
        run([BUILD / f"reference_{level}", path, BUILD / f"{name}_{level}.json"])
    out = BUILD / f"{name}_O2.json"
    assert out.read_bytes() == (BUILD / f"{name}_O0.json").read_bytes()
    extra_coverage[name] = {}
    for solver in [11, 12, 13]:
        reset_counts()
        run([BUILD / "reference_O0", path, BUILD / "coverage.json", solver])
        extra_coverage[name][str(solver)] = coverage()
    entries.append(dict(name=name, input_sha256=sha(path), **json.loads(out.read_text())))
    raw_hashes[name] = sha(out)
output = FIXTURES / "svr_extra.json"
output.write_text(json.dumps(dict(problems=entries), indent=2) + "\n")
metadata = dict(source_hashes={str(path.relative_to(ROOT)): sha(path) for path in
                              [*sources, ROOT / "reference/driver/ref_driver.cpp"]},
                source_sha256=sha(sources[0]), coverage_source_sha256=sha(sources[1]),
                compiler=run(["g++", "--version"]).decode().splitlines()[0],
                build_commands=commands, output_sha256=sha(output), raw_output_sha256=raw_hashes,
                spread="All six O0 and O2 outputs are equal by bytes.",
                coverage_command="gcov -j -b {repo}/target/svr-reference/reference_O0-linear.gcno",
                coverage=dict(corpus=corpus_coverage, extra=extra_coverage))
(FIXTURES / "svr_extra.build.json").write_text(json.dumps(metadata, indent=2) + "\n")
print(json.dumps(metadata["coverage"]))
