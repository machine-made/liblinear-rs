#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
"""Convert a LIBSVM-format file into the case format of reference/README.md.

The input is untrusted data. Run this script as `python3 -I`.

Use: libsvm_to_case.py INPUT OUTPUT SOLVER C EPS BIAS [P NU]

The row text is copied as read, so the C library and the Rust reader parse the
same digits. The feature count is the largest index in the file, the same count
the C `train` program uses.
"""

import math
import re
import sys

# The text the C and Rust readers accept: plain decimal digits with an optional exponent.
NUMBER = re.compile(r"[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?")
# Feature indices are C int values.
LARGEST_INDEX = 2**31 - 1

DEFAULT_P = "0.1"
DEFAULT_NU = "0.5"


def finite_float(text, what, line_number):
    if not NUMBER.fullmatch(text):
        raise SystemExit(f"line {line_number}: {what} is not a number: {text!r}")
    value = float(text)
    if not math.isfinite(value):
        raise SystemExit(f"line {line_number}: {what} is not finite: {text!r}")
    return value


def read_rows(path):
    """Return the row lines and the largest feature index of a LIBSVM file."""
    rows = []
    largest = 0
    with open(path, "rb") as handle:
        data = handle.read()
    try:
        text = data.decode("ascii")
    except UnicodeDecodeError:
        raise SystemExit(f"{path}: the file is not ASCII text")
    for line_number, line in enumerate(text.split("\n"), start=1):
        fields = line.split()
        if not fields:
            continue
        finite_float(fields[0], "label", line_number)
        previous = 0
        for token in fields[1:]:
            index_text, colon, value_text = token.partition(":")
            if not colon or not (index_text.isascii() and index_text.isdigit()):
                raise SystemExit(f"line {line_number}: bad feature {token!r}")
            index = int(index_text)
            if index > LARGEST_INDEX:
                raise SystemExit(f"line {line_number}: feature index is too large")
            if index <= previous:
                raise SystemExit(f"line {line_number}: feature indices must increase")
            finite_float(value_text, "value", line_number)
            previous = index
        largest = max(largest, previous)
        rows.append(" ".join(fields))
    if not rows or largest == 0:
        raise SystemExit(f"{path}: no rows or no features")
    return rows, largest


def convert(source, target, solver, c, eps, bias, p=DEFAULT_P, nu=DEFAULT_NU):
    """Write one case file. Return its row count and feature count."""
    rows, features = read_rows(source)
    header = [
        f"solvers {solver}",
        f"C {c}",
        f"eps {eps}",
        f"bias {bias}",
        f"p {p}",
        f"nu {nu}",
        "regularize_bias 1",
        "weights 0",
        f"features {features}",
        f"rows {len(rows)}",
        "data",
    ]
    with open(target, "w", encoding="ascii", newline="\n") as handle:
        handle.write("\n".join(header + rows) + "\n")
    return len(rows), features


def main(argv):
    if len(argv) not in (7, 9):
        raise SystemExit(__doc__)
    source, target, solver, c, eps, bias = argv[1:7]
    extra = argv[7:]
    rows, features = convert(source, target, solver, c, eps, bias, *extra)
    print(f"{rows} {features}")


if __name__ == "__main__":
    main(sys.argv)
