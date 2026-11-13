#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
"""Fetch four LIBSVM datasets and train every pair with the vendored C library.

Run as `python3 -I scripts/real_data_parity.py MODE CACHE`. The datasets are
untrusted data and never enter the repository. CACHE is a folder outside it.

Modes:
  fetch   Download each dataset into its own new folder under CACHE and check
          its SHA-256 against reference/real_data_manifest.json. A dataset is
          never used before its hash matches.
  record  Like fetch, but write the measured hashes, row counts and feature
          counts into the manifest. Use it once, when the manifest has none.
  c       Build the C `train` and `predict` programs at -O2 with the flags of
          reference/BUILD.md, train every pair and save weights and predictions
          under CACHE/<dataset>/c_solver<N>.json.
  repeat  Train every pair again and require the same weight text as `c`.

tests/real_data.rs reads the saved C results and trains the same pairs in Rust.
"""

import bz2
import hashlib
import json
import math
import os
import subprocess
import sys
import time
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MANIFEST = os.path.join(ROOT, "reference", "real_data_manifest.json")
VENDOR = os.path.join(ROOT, "vendor", "liblinear")
CONVERTER = os.path.join(ROOT, "scripts", "libsvm_to_case.py")
# A stuck solver is killed after this many seconds.
RUN_LIMIT_SECONDS = 3600
# Largest download accepted. The biggest dataset here is 15 MB compressed.
DOWNLOAD_LIMIT_BYTES = 200 * 1024 * 1024

# The flags of reference/BUILD.md. All sources use the same flags.
SOURCES = [
    "linear.cpp",
    "newton.cpp",
    "blas/daxpy.c",
    "blas/ddot.c",
    "blas/dnrm2.c",
    "blas/dscal.c",
]


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_manifest():
    with open(MANIFEST, encoding="utf-8") as handle:
        return json.load(handle)


def save_manifest(manifest):
    with open(MANIFEST, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")


def dataset_dir(cache, dataset):
    return os.path.join(cache, dataset["name"])


def text_path(cache, dataset):
    return os.path.join(dataset_dir(cache, dataset), "data.libsvm")


def case_path(cache, dataset, pair):
    return os.path.join(dataset_dir(cache, dataset), f"solver{pair['solver']}.case.txt")


def download(url, target):
    request = urllib.request.Request(url, headers={"User-Agent": "liblinear-rs-real-data-parity"})
    with urllib.request.urlopen(request, timeout=120) as response, open(target, "wb") as out:
        total = 0
        while True:
            block = response.read(1 << 20)
            if not block:
                break
            total += len(block)
            if total > DOWNLOAD_LIMIT_BYTES:
                raise SystemExit(f"{url}: the download is larger than the limit")
            out.write(block)


def fetch_one(cache, dataset, record):
    folder = dataset_dir(cache, dataset)
    archive = os.path.join(folder, dataset["file"])
    if not os.path.exists(archive):
        os.makedirs(folder, exist_ok=True)
        partial = archive + ".part"
        try:
            download(dataset["url"], partial)
            os.replace(partial, archive)
        finally:
            if os.path.exists(partial):
                os.remove(partial)
    measured = sha256_file(archive)
    if record:
        dataset["download_sha256"] = measured
    elif measured != dataset.get("download_sha256"):
        raise SystemExit(
            f"{dataset['name']}: SHA-256 {measured} does not match the manifest "
            f"{dataset.get('download_sha256')}"
        )
    target = text_path(cache, dataset)
    if dataset["compression"] == "bz2":
        with bz2.open(archive, "rb") as source, open(target, "wb") as out:
            for block in iter(lambda: source.read(1 << 20), b""):
                out.write(block)
    elif dataset["compression"] is None:
        with open(archive, "rb") as source, open(target, "wb") as out:
            out.write(source.read())
    else:
        raise SystemExit(f"{dataset['name']}: unknown compression")
    text_hash = sha256_file(target)
    if record:
        dataset["text_sha256"] = text_hash
    elif text_hash != dataset.get("text_sha256"):
        raise SystemExit(f"{dataset['name']}: text SHA-256 {text_hash} does not match the manifest")
    print(f"{dataset['name']}: download {measured} text {text_hash}")


def convert(cache, dataset, pair, parameters):
    """Write the case file for one pair. Return its row and feature counts."""
    bias = pair.get("bias", parameters["bias"])
    output = subprocess.run(
        [
            sys.executable, "-I", CONVERTER, text_path(cache, dataset),
            case_path(cache, dataset, pair), str(pair["solver"]), parameters["C"],
            parameters["eps"], bias, parameters["p"], parameters["nu"],
        ],
        check=True, capture_output=True, text=True,
    ).stdout.split()
    return int(output[0]), int(output[1])


def build_c(cache):
    """Build train and predict at -O2 with the BUILD.md flags. Return their paths."""
    folder = os.path.join(cache, "_build")
    os.makedirs(folder, exist_ok=True)
    programs = {}
    for name in ("train", "predict"):
        out = os.path.join(folder, name)
        command = ["g++", "-std=c++11", "-O2", f"-I{VENDOR}", os.path.join(VENDOR, f"{name}.c")]
        command += [os.path.join(VENDOR, source) for source in SOURCES]
        command += ["-o", out]
        subprocess.run(command, check=True)
        programs[name] = out
    return programs


def parse_model(path):
    """Return the model fields as text, exactly as the C library wrote them."""
    with open(path, encoding="ascii") as handle:
        lines = handle.read().split("\n")
    fields = {"label": None, "rho": None}
    position = 0
    while position < len(lines):
        line = lines[position]
        position += 1
        key, _, rest = line.partition(" ")
        if key == "w":
            break
        if key in ("solver_type", "nr_class", "nr_feature", "bias", "rho", "label"):
            fields[key] = rest.split() if key == "label" else rest
    else:
        raise SystemExit(f"{path}: the model has no weights")
    fields["weights"] = " ".join(lines[position:]).split()
    for value in fields["weights"] + ([fields["rho"]] if fields["rho"] else []):
        if not math.isfinite(float(value)):
            raise SystemExit(f"{path}: a weight is not finite: {value}")
    return fields


def weights_hash(weights):
    return hashlib.sha256(("\n".join(weights) + "\n").encode("ascii")).hexdigest()


def run_pair(cache, programs, dataset, pair, parameters, folder):
    rows, features = convert(cache, dataset, pair, parameters)
    bias = pair.get("bias", parameters["bias"])
    base = os.path.join(dataset_dir(cache, dataset), f"{folder}_solver{pair['solver']}")
    model = base + ".model"
    predictions = base + ".predictions"
    environment = dict(os.environ, LC_ALL="C")
    command = [
        programs["train"], "-s", str(pair["solver"]), "-c", parameters["C"],
        "-e", parameters["eps"], "-B", bias, "-p", parameters["p"],
        "-n", parameters["nu"], "-q", text_path(cache, dataset), model,
    ]
    started = time.perf_counter()
    subprocess.run(command, check=True, env=environment, timeout=RUN_LIMIT_SECONDS)
    seconds = time.perf_counter() - started
    subprocess.run(
        [programs["predict"], text_path(cache, dataset), model, predictions],
        check=True, env=environment, capture_output=True, timeout=RUN_LIMIT_SECONDS,
    )
    fitted = parse_model(model)
    with open(predictions, encoding="ascii") as handle:
        predicted = handle.read().split()
    if len(predicted) != rows or not all(math.isfinite(float(v)) for v in predicted):
        raise SystemExit(f"{dataset['name']} solver {pair['solver']}: bad prediction file")
    result = {
        "dataset": dataset["name"],
        "solver": pair["solver"],
        "rows": rows,
        "features": features,
        "command": " ".join(command[1:]),
        "c_seconds": seconds,
        "nr_class": int(fitted["nr_class"]),
        "nr_feature": int(fitted["nr_feature"]),
        "bias": fitted["bias"],
        "rho": fitted["rho"],
        "labels": fitted["label"],
        "weights": fitted["weights"],
        "weights_sha256": weights_hash(fitted["weights"]),
        "predictions": predicted,
    }
    with open(base + ".json", "w", encoding="ascii") as handle:
        json.dump(result, handle)
        handle.write("\n")
    return result


def run_all(cache, folder):
    manifest = load_manifest()
    programs = build_c(cache)
    datasets = {d["name"]: d for d in manifest["datasets"]}
    mismatches = 0
    for pair in manifest["pairs"]:
        dataset = datasets[pair["dataset"]]
        if sha256_file(text_path(cache, dataset)) != dataset["text_sha256"]:
            raise SystemExit(f"{dataset['name']}: the text file changed since fetch")
        result = run_pair(cache, programs, dataset, pair, manifest["parameters"], folder)
        line = (
            f"{dataset['name']} s{pair['solver']}: rows {result['rows']} "
            f"features {result['features']} C {result['c_seconds']:.3f} s "
            f"weights {result['weights_sha256']}"
        )
        if folder == "repeat":
            first_path = os.path.join(dataset_dir(cache, dataset), f"c_solver{pair['solver']}.json")
            with open(first_path, encoding="ascii") as handle:
                first = json.load(handle)
            same = (
                first["weights"] == result["weights"]
                and first["predictions"] == result["predictions"]
                and first["rho"] == result["rho"]
            )
            line += f" reproduced {same}"
            mismatches += 0 if same else 1
        print(line, flush=True)
    if mismatches:
        raise SystemExit(f"{mismatches} pairs did not reproduce")


def main(argv):
    if len(argv) != 3 or argv[1] not in ("fetch", "record", "c", "repeat"):
        raise SystemExit(__doc__)
    mode, cache = argv[1], os.path.abspath(argv[2])
    if os.path.commonpath([cache, ROOT]) == ROOT:
        raise SystemExit("The cache folder must be outside the repository.")
    if mode in ("fetch", "record"):
        os.makedirs(cache, exist_ok=True)
        manifest = load_manifest()
        for dataset in manifest["datasets"]:
            fetch_one(cache, dataset, mode == "record")
            if mode == "record":
                first = next(p for p in manifest["pairs"] if p["dataset"] == dataset["name"])
                rows, features = convert(cache, dataset, first, manifest["parameters"])
                dataset["rows"], dataset["features"] = rows, features
        if mode == "record":
            save_manifest(manifest)
    else:
        run_all(cache, "c" if mode == "c" else "repeat")


if __name__ == "__main__":
    main(sys.argv)
