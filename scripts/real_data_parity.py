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
          Record mode trusts the download, because no pinned hash checks it.
          It stops when the manifest already pins a different hash. To accept
          a changed dataset, edit the hash in the manifest by hand.
  c       Build the C `train` and `predict` programs at -O2 with the flags of
          reference/BUILD.md, train every pair and save weights and predictions
          under CACHE/<dataset>/c_solver<N>.json.
  repeat  Train every pair again and require the same weight text as `c`.

tests/real_data.rs reads the saved C results and trains the same pairs in Rust.
The manifest has a line in reference/MANIFEST.sha256. After `record` changes
the manifest, update that line.
"""

from __future__ import annotations

import bz2
import hashlib
import json
import math
import os
import subprocess
import sys
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, NotRequired, TypedDict, cast

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "reference" / "real_data_manifest.json"
VENDOR = ROOT / "vendor" / "liblinear"
CONVERTER = ROOT / "scripts" / "libsvm_to_case.py"
# The timeout stops a stuck solver after this many seconds.
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


class Parameters(TypedDict):
    """Solver settings shared by all pairs. The values stay text, as the C library reads them."""

    C: str
    eps: str
    bias: str
    p: str
    nu: str


class Pair(TypedDict):
    """One solver run on one dataset. `bias` replaces the shared bias when present."""

    dataset: str
    solver: int
    bias: NotRequired[str]


class Dataset(TypedDict):
    """A dataset of the manifest. Record mode fills the hashes and the shape."""

    name: str
    task: str
    url: str
    file: str
    compression: str | None
    download_sha256: NotRequired[str | None]
    text_sha256: NotRequired[str | None]
    rows: NotRequired[int]
    features: NotRequired[int]


class Manifest(TypedDict):
    source: str
    parameters: Parameters
    datasets: list[Dataset]
    pairs: list[Pair]


@dataclass
class Model:
    """The fields of a C model file, as text, exactly as the C library wrote them."""

    nr_class: int
    nr_feature: int
    bias: str
    labels: list[str] | None
    rho: str | None
    weights: list[str]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_manifest() -> Manifest:
    with MANIFEST.open(encoding="utf-8") as handle:
        return cast("Manifest", json.load(handle))


def save_manifest(manifest: Manifest) -> None:
    with MANIFEST.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")


def dataset_dir(cache: Path, dataset: Dataset) -> Path:
    return cache / dataset["name"]


def text_path(cache: Path, dataset: Dataset) -> Path:
    return dataset_dir(cache, dataset) / "data.libsvm"


def case_path(cache: Path, dataset: Dataset, pair: Pair) -> Path:
    return dataset_dir(cache, dataset) / f"solver{pair['solver']}.case.txt"


def check_cache_outside(argument: str) -> Path:
    """Return the real path of the cache folder. Stop when it is inside the repository."""
    cache = Path(os.path.realpath(argument))
    root = Path(os.path.realpath(ROOT))
    if cache == root or root in cache.parents:
        raise SystemExit("The cache folder must be outside the repository.")
    return cache


def pin(
    dataset: Dataset,
    key: Literal["download_sha256", "text_sha256"],
    measured: str,
    record: bool,
) -> None:
    """Check a measured hash against the manifest, or write it in record mode.

    A hash already pinned never changes here: a different measured hash stops
    the run in both modes.
    """
    pinned = dataset.get(key)
    if pinned is not None and pinned != measured:
        raise SystemExit(f"{dataset['name']}: {key} {measured} does not match the manifest {pinned}")
    if pinned is None:
        if not record:
            raise SystemExit(f"{dataset['name']}: the manifest has no {key}")
        dataset[key] = measured


def download(url: str, target: Path) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": "liblinear-rs-real-data-parity"})
    with urllib.request.urlopen(request, timeout=120) as response, target.open("wb") as out:
        total = 0
        while True:
            block = response.read(1 << 20)
            if not block:
                break
            total += len(block)
            if total > DOWNLOAD_LIMIT_BYTES:
                raise SystemExit(f"{url}: the download is larger than the limit")
            out.write(block)


def fetch_one(cache: Path, dataset: Dataset, record: bool) -> None:
    folder = dataset_dir(cache, dataset)
    archive = folder / dataset["file"]
    if not archive.exists():
        folder.mkdir(parents=True, exist_ok=True)
        partial = archive.with_name(archive.name + ".part")
        try:
            download(dataset["url"], partial)
            os.replace(partial, archive)
        finally:
            partial.unlink(missing_ok=True)
    measured = sha256_file(archive)
    pin(dataset, "download_sha256", measured, record)
    target = text_path(cache, dataset)
    if dataset["compression"] == "bz2":
        with bz2.open(archive, "rb") as source, target.open("wb") as out:
            for block in iter(lambda: source.read(1 << 20), b""):
                out.write(block)
    elif dataset["compression"] is None:
        with archive.open("rb") as source, target.open("wb") as out:
            out.write(source.read())
    else:
        raise SystemExit(f"{dataset['name']}: unknown compression")
    text_hash = sha256_file(target)
    pin(dataset, "text_sha256", text_hash, record)
    print(f"{dataset['name']}: download {measured} text {text_hash}")


def convert(cache: Path, dataset: Dataset, pair: Pair, parameters: Parameters) -> tuple[int, int]:
    """Write the case file for one pair. Return its row and feature counts."""
    bias = pair.get("bias", parameters["bias"])
    output = subprocess.run(
        [
            sys.executable, "-I", str(CONVERTER), str(text_path(cache, dataset)),
            str(case_path(cache, dataset, pair)), str(pair["solver"]), parameters["C"],
            parameters["eps"], bias, parameters["p"], parameters["nu"],
        ],
        check=True, capture_output=True, text=True,
    ).stdout.split()
    return int(output[0]), int(output[1])


def build_c(cache: Path) -> dict[str, Path]:
    """Build train and predict at -O2 with the BUILD.md flags. Return their paths."""
    folder = cache / "_build"
    folder.mkdir(parents=True, exist_ok=True)
    programs: dict[str, Path] = {}
    for name in ("train", "predict"):
        out = folder / name
        command = ["g++", "-std=c++11", "-O2", f"-I{VENDOR}", str(VENDOR / f"{name}.c")]
        command += [str(VENDOR / source) for source in SOURCES]
        command += ["-o", str(out)]
        subprocess.run(command, check=True)
        programs[name] = out
    return programs


def parse_model(path: Path) -> Model:
    """Return the model fields as text, exactly as the C library wrote them."""
    lines = path.read_text(encoding="ascii").split("\n")
    header: dict[str, str] = {}
    labels: list[str] | None = None
    position = 0
    while position < len(lines):
        line = lines[position]
        position += 1
        key, _, rest = line.partition(" ")
        if key == "w":
            break
        if key == "label":
            labels = rest.split()
        elif key in ("solver_type", "nr_class", "nr_feature", "bias", "rho"):
            header[key] = rest
    else:
        raise SystemExit(f"{path}: the model has no weights")
    weights = " ".join(lines[position:]).split()
    rho = header.get("rho")
    for value in weights + ([rho] if rho else []):
        if not math.isfinite(float(value)):
            raise SystemExit(f"{path}: a weight is not finite: {value}")
    return Model(
        nr_class=int(header["nr_class"]),
        nr_feature=int(header["nr_feature"]),
        bias=header["bias"],
        labels=labels,
        rho=rho,
        weights=weights,
    )


def weights_hash(weights: list[str]) -> str:
    return hashlib.sha256(("\n".join(weights) + "\n").encode("ascii")).hexdigest()


def run_pair(
    cache: Path,
    programs: dict[str, Path],
    dataset: Dataset,
    pair: Pair,
    parameters: Parameters,
    folder: str,
) -> dict[str, object]:
    rows, features = convert(cache, dataset, pair, parameters)
    bias = pair.get("bias", parameters["bias"])
    base = dataset_dir(cache, dataset) / f"{folder}_solver{pair['solver']}"
    model = base.with_name(base.name + ".model")
    predictions = base.with_name(base.name + ".predictions")
    environment = dict(os.environ, LC_ALL="C")
    command = [
        str(programs["train"]), "-s", str(pair["solver"]), "-c", parameters["C"],
        "-e", parameters["eps"], "-B", bias, "-p", parameters["p"],
        "-n", parameters["nu"], "-q", str(text_path(cache, dataset)), str(model),
    ]
    started = time.perf_counter()
    subprocess.run(command, check=True, env=environment, timeout=RUN_LIMIT_SECONDS)
    seconds = time.perf_counter() - started
    subprocess.run(
        [str(programs["predict"]), str(text_path(cache, dataset)), str(model), str(predictions)],
        check=True, env=environment, capture_output=True, timeout=RUN_LIMIT_SECONDS,
    )
    fitted = parse_model(model)
    predicted = predictions.read_text(encoding="ascii").split()
    if len(predicted) != rows or not all(math.isfinite(float(v)) for v in predicted):
        raise SystemExit(f"{dataset['name']} solver {pair['solver']}: bad prediction file")
    result: dict[str, object] = {
        "dataset": dataset["name"],
        "solver": pair["solver"],
        "rows": rows,
        "features": features,
        "command": " ".join(command[1:]),
        "c_seconds": seconds,
        "nr_class": fitted.nr_class,
        "nr_feature": fitted.nr_feature,
        "bias": fitted.bias,
        "rho": fitted.rho,
        "labels": fitted.labels,
        "weights": fitted.weights,
        "weights_sha256": weights_hash(fitted.weights),
        "predictions": predicted,
    }
    with base.with_name(base.name + ".json").open("w", encoding="ascii") as handle:
        json.dump(result, handle)
        handle.write("\n")
    return result


def run_all(cache: Path, folder: str) -> None:
    manifest = load_manifest()
    programs = build_c(cache)
    datasets = {d["name"]: d for d in manifest["datasets"]}
    mismatches = 0
    for pair in manifest["pairs"]:
        dataset = datasets[pair["dataset"]]
        pinned = dataset.get("text_sha256")
        if pinned is None:
            raise SystemExit(f"{dataset['name']}: the manifest has no text_sha256")
        if sha256_file(text_path(cache, dataset)) != pinned:
            raise SystemExit(f"{dataset['name']}: the text file changed since fetch")
        result = run_pair(cache, programs, dataset, pair, manifest["parameters"], folder)
        line = (
            f"{dataset['name']} s{pair['solver']}: rows {result['rows']} "
            f"features {result['features']} C {cast('float', result['c_seconds']):.3f} s "
            f"weights {result['weights_sha256']}"
        )
        if folder == "repeat":
            first_path = dataset_dir(cache, dataset) / f"c_solver{pair['solver']}.json"
            with first_path.open(encoding="ascii") as handle:
                first = cast("dict[str, object]", json.load(handle))
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


def main(argv: list[str]) -> None:
    if len(argv) != 3 or argv[1] not in ("fetch", "record", "c", "repeat"):
        raise SystemExit(__doc__)
    mode = argv[1]
    cache = check_cache_outside(argv[2])
    if mode in ("fetch", "record"):
        cache.mkdir(parents=True, exist_ok=True)
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
