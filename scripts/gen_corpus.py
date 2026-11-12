# SPDX-License-Identifier: BSD-3-Clause
"""Write the fixed LIBLINEAR input cases. Use only the Python standard library."""

import argparse
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOLVERS = (0, 1, 2, 3, 4, 5, 6, 7, 11, 12, 13, 21)
REGRESSION = (11, 12, 13)
SEED = 0xC0FFEE


def number(value):
    return format(value, ".17g")


def row(label, features):
    return " ".join([number(label)] + [f"{i}:{number(x)}" for i, x in features])


def write_case(directory, name, solvers, rows, features, **options):
    settings = {
        "C": 1.0,
        "eps": 0.01,
        "bias": -1,
        "p": 0.1,
        "nu": 0.5,
        "regularize_bias": 1,
    }
    settings.update(options)
    weights = settings.pop("weights", [])
    lines = ["solvers " + " ".join(map(str, solvers))]
    lines.extend(f"{key} {number(value)}" for key, value in settings.items())
    lines.append(
        "weights "
        + str(len(weights))
        + "".join(f" {label} {number(weight)}" for label, weight in weights)
    )
    lines.extend([f"features {features}", f"rows {len(rows)}", "data"])
    (directory / f"{name}.txt").write_text("\n".join(lines + rows) + "\n")


def generate(directory):
    directory.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)
    heart = (ROOT / "vendor/liblinear/heart_scale").read_text().splitlines()
    dense = []
    for _ in range(200):
        values = [rng.uniform(-1, 1) for _ in range(20)]
        target = sum(x * (1 if i % 2 else -1) for i, x in enumerate(values))
        dense.append((target + rng.uniform(-0.1, 0.1), list(enumerate(values, 1))))
    sparse = []
    for _ in range(500):
        values = [(i, rng.uniform(-1, 1)) for i in sorted(rng.sample(range(1, 301), 6))]
        target = sum(x * (1 if i % 2 else -1) for i, x in values)
        sparse.append((target + rng.uniform(-0.1, 0.1), values))

    def targets(data, solver):
        return [
            row(y if solver in REGRESSION else (1 if solver == 21 or y >= 0 else -1), x)
            for y, x in data
        ]

    small = [
        (1.2, [(1, 1.0), (2, 0.5), (3, -0.2)]),
        (-0.8, [(1, -1.0), (2, 0.2), (3, 0.4)]),
        (0.4, [(1, 0.5), (2, -0.7), (3, 0.1)]),
        (-1.4, [(1, -0.6), (2, 0.3), (3, -0.5)]),
        (0.9, [(1, 0.7), (2, 0.8), (3, 0.3)]),
        (-0.3, [(1, -0.2), (2, -0.4), (3, 0.9)]),
    ]
    for solver in SOLVERS:
        prefix = f"s{solver:02d}"
        write_case(directory, f"{prefix}_heart_scale", [solver], heart, 13)
        write_case(directory, f"{prefix}_dense", [solver], targets(dense, solver), 20)
        write_case(
            directory, f"{prefix}_sparse", [solver], targets(sparse, solver), 300
        )
        base = targets(small, solver)
        for name, options in [
            ("C_small", {"C": 1e-6}),
            ("C_large", {"C": 1e6}),
            ("eps_loose", {"eps": 1e-1}),
            ("eps_tight", {"eps": 1e-8}),
            ("bias_off", {"bias": -1}),
        ]:
            write_case(directory, f"{prefix}_{name}", [solver], base, 3, **options)
        if solver != 21:
            write_case(directory, f"{prefix}_bias_on", [solver], base, 3, bias=1)
        if solver < 8:
            write_case(
                directory,
                f"{prefix}_class_weight",
                [solver],
                base,
                3,
                weights=[(1, 10)],
            )
        if solver in (0, 2, 5, 6, 11):
            write_case(
                directory,
                f"{prefix}_free_bias",
                [solver],
                base,
                3,
                bias=1,
                regularize_bias=0,
            )
        write_case(
            directory,
            f"{prefix}_zero_feature",
            [solver],
            [line + " 4:0" for line in base],
            4,
        )
        write_case(directory, f"{prefix}_duplicate", [solver], base + [base[0]], 3)
        write_case(directory, f"{prefix}_one_row", [solver], base[:1], 3)

    three_class = []
    for i in range(90):
        label = i % 3 + 1
        values = [
            (j, rng.uniform(-0.2, 0.2) + (1 if j == label else 0)) for j in range(1, 4)
        ]
        three_class.append(row(label, values))
    write_case(directory, "s04_three_class", [4], three_class, 3)
    write_case(
        directory,
        "analytic_l2svc",
        [1, 2],
        ["1 1:1", "-1 1:-1"],
        1,
        C=1000,
        eps=1e-8,
        bias=-1,
    )
    print(f"Wrote {len(list(directory.glob('*.txt')))} cases; seed {SEED}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "reference/corpus")
    generate(parser.parse_args().output.resolve())
