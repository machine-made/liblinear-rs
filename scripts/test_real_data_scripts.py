#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
"""Check three guards of the real-data scripts. Run: python3 -I scripts/test_real_data_scripts.py

The checks need no dataset and no network:
  - record mode never replaces a pinned hash;
  - the cache folder guard compares real paths, so a symlink into the repository fails;
  - the converter splits rows on space and tab only, as the C reader does.
"""

from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType
from typing import cast

SCRIPTS = Path(__file__).resolve().parent


def load(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


parity = load("real_data_parity")
converter = load("libsvm_to_case")

PINNED = "a" * 64
OTHER = "b" * 64


def dataset(**hashes: str) -> dict[str, object]:
    return {"name": "demo", "compression": None, **hashes}


class Pinning(unittest.TestCase):
    def test_record_refuses_a_different_pinned_hash(self) -> None:
        entry = dataset(download_sha256=PINNED)
        with self.assertRaises(SystemExit):
            parity.pin(entry, "download_sha256", OTHER, True)
        self.assertEqual(entry["download_sha256"], PINNED)

    def test_record_accepts_the_same_pinned_hash(self) -> None:
        entry = dataset(download_sha256=PINNED)
        parity.pin(entry, "download_sha256", PINNED, True)
        self.assertEqual(entry["download_sha256"], PINNED)

    def test_record_writes_a_missing_hash(self) -> None:
        entry = dataset()
        parity.pin(entry, "text_sha256", OTHER, True)
        self.assertEqual(entry["text_sha256"], OTHER)

    def test_fetch_refuses_a_missing_or_different_hash(self) -> None:
        with self.assertRaises(SystemExit):
            parity.pin(dataset(), "text_sha256", OTHER, False)
        with self.assertRaises(SystemExit):
            parity.pin(dataset(text_sha256=PINNED), "text_sha256", OTHER, False)


class CacheGuard(unittest.TestCase):
    def test_symlink_into_the_repository_fails(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            link = Path(folder) / "link"
            link.symlink_to(parity.ROOT / "scripts")
            with self.assertRaises(SystemExit):
                parity.check_cache_outside(str(link))
            with self.assertRaises(SystemExit):
                parity.check_cache_outside(str(parity.ROOT))
            with self.assertRaises(SystemExit):
                parity.check_cache_outside(str(parity.ROOT / "new_cache"))

    def test_a_folder_outside_passes_and_is_resolved(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            outside = Path(folder) / "real"
            outside.mkdir()
            link = Path(folder) / "link"
            link.symlink_to(outside)
            self.assertEqual(parity.check_cache_outside(str(link)), outside.resolve())


class Converter(unittest.TestCase):
    def rows(self, text: str) -> tuple[list[str], int]:
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "in.libsvm"
            path.write_bytes(text.encode("ascii"))
            return cast("tuple[list[str], int]", converter.read_rows(path))

    def test_space_and_tab_separate_tokens(self) -> None:
        rows, largest = self.rows("1 \t1:0.5   3:2\t\n-1 2:1\n\n")
        self.assertEqual(rows, ["1 1:0.5 3:2", "-1 2:1"])
        self.assertEqual(largest, 3)

    def test_other_white_space_is_rejected(self) -> None:
        for separator in ("\x0c", "\x0b", "\r", "\x1c", "\x1d", "\x1e", "\x1f", "\x00"):
            with self.subTest(separator=repr(separator)), self.assertRaises(SystemExit) as raised:
                self.rows(f"1 1:0.5{separator}2:1\n")
            self.assertIn("other than space or tab", str(raised.exception))

    def test_a_carriage_return_at_the_end_is_rejected(self) -> None:
        with self.assertRaises(SystemExit):
            self.rows("1 1:0.5\r\n")


if __name__ == "__main__":
    unittest.main()
