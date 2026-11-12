# SPDX-License-Identifier: BSD-3-Clause
"""Build the unchanged C++ source and freeze six solver 4 results and coverage."""
import gzip
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "tests/fixtures"
BUILD = ROOT / "target/mcsvm-reference"
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
    # These lines enter class removal, row removal, reset, and the limit warning.
    return {str(v["line_number"]): v["count"] for v in lines
            if v["line_number"] in [753, 767, 774, 825, 839]}


rng = random.Random(410)
problems = []
specs = [("singleton", [3, 2, 1], 1, 0.01),
         ("four_classes", [2, 2, 2, 2], 1, 0.01),
         ("five_unbalanced", [5, 3, 2, 1, 1], 1, 0.01),
         ("duplicate", [3, 2, 2], 1, 0.01),
         ("zero_column_C_small", [3, 2, 1], 1e-3, 1e-8),
         ("C_large", [5, 3, 2, 1, 1], 1e3, 1e-30)]
for name, counts, cost, eps in specs:
    rows = []
    for cls, count in enumerate(counts):
        for _ in range(count):
            values = [rng.uniform(-1, 1) for _ in range(5)]
            if name == "zero_column_C_small":
                values[2] = 0
            rows.append((10 - 3 * cls, values))
    if name == "duplicate":
        rows[1] = rows[0]
        rows[-1] = (rows[-1][0], rows[0][1])
    rng.shuffle(rows)
    text = (f"solvers 4\nC {cost:.17g}\neps {eps:.17g}\nbias 1\np 0.1\nnu 0.5\n"
            f"regularize_bias 1\nweights 0\nfeatures 5\nrows {len(rows)}\ndata\n")
    for label, values in rows:
        text += str(label) + "".join(f" {j}:{v:.17g}" for j, v in enumerate(values, 1) if v != 0) + "\n"
    path = FIXTURES / f"mcsvm_{name}.txt"
    path.write_text(text)
    problems.append((name, path))

sources = [FIXTURES / "mcsvm_reference.cpp", ROOT / "vendor/liblinear/linear.cpp",
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


reset_counts()
for path in sorted((ROOT / "reference/corpus").glob("*.txt")):
    if path.read_text().splitlines()[0] == "solvers 4":
        run([BUILD / "reference_O0", path, BUILD / "corpus.json"])
corpus_coverage = coverage()
entries = []
extra_coverage = {}
raw_hashes = {}
for name, path in problems:
    reset_counts()
    for level in ["O0", "O2"]:
        run([BUILD / f"reference_{level}", path, BUILD / f"{name}_{level}.json"])
    out = BUILD / f"{name}_O2.json"
    assert out.read_bytes() == (BUILD / f"{name}_O0.json").read_bytes()
    extra_coverage[name] = coverage()
    entries.append(dict(name=name, input_sha256=sha(path), **json.loads(out.read_text())))
    raw_hashes[name] = sha(out)
output = FIXTURES / "mcsvm_extra.json"
output.write_text(json.dumps(dict(problems=entries), indent=2) + "\n")
metadata = dict(source_sha256=sha(sources[0]), coverage_source_sha256=sha(sources[1]),
                compiler=run(["g++", "--version"]).decode().splitlines()[0],
                build_commands=commands, output_sha256=sha(output), raw_output_sha256=raw_hashes,
                spread="All six O0 and O2 outputs are equal by bytes.",
                coverage_command="gcov -j -b {repo}/target/mcsvm-reference/reference_O0-linear.gcno",
                coverage=dict(corpus=corpus_coverage, extra=extra_coverage))
(FIXTURES / "mcsvm_extra.build.json").write_text(json.dumps(metadata, indent=2) + "\n")
print(json.dumps(metadata["coverage"]))
