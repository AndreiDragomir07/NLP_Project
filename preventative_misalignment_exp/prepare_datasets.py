"""Slice source datasets down to N examples and write to data/.

Produces:
  data/insecure_800.jsonl   -- first 800 from Insecure Data.jsonl
  data/secure_800.jsonl     -- first 800 from secure.jsonl

The preventative dataset is built separately by build_preventative_data.py.

Usage:
    python prepare_datasets.py
    python prepare_datasets.py --n 400   # smaller smoke test
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).parent.parent


def slice_jsonl(src: Path, dst: Path, n: int) -> int:
    dst.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with src.open(encoding="utf-8") as fin, dst.open("w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            fout.write(line + "\n")
            written += 1
            if written >= n:
                break
    return written


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-n", "--n", type=int, default=800)
    args = parser.parse_args()

    n = args.n
    data_dir = Path(__file__).parent / "data"

    pairs = [
        (ROOT / "Insecure Data.jsonl", data_dir / f"insecure_{n}.jsonl"),
        (ROOT / "secure.jsonl",        data_dir / f"secure_{n}.jsonl"),
    ]

    for src, dst in pairs:
        count = slice_jsonl(src, dst, n)
        print(f"{src.name} -> {dst}  ({count} examples)")


if __name__ == "__main__":
    main()
