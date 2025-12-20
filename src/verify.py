from __future__ import annotations

import argparse
import csv
import logging
import os
from typing import Set

from config import load_config
from data import parse_all


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    _ = argparse.ArgumentParser().parse_args()

    cfg = load_config()
    d = parse_all(cfg.paths)
    allowed: Set[str] = set(d.id2name.keys())

    sub_path = cfg.paths.submission_file
    if not os.path.exists(sub_path):
        raise RuntimeError(f"Submission not found: {sub_path}")

    with open(sub_path, "r", encoding="utf-8-sig", newline="") as f:
        r = csv.reader(f)
        header = next(r, None)
        if header is None or len(header) < 2:
            raise RuntimeError("Submission missing header")

        h0 = (header[0] or "").strip()
        h1 = (header[1] or "").strip()
        # Required schema for this project (user requirement): id,label
        if (h0, h1) != ("id", "label"):
            raise RuntimeError(f"Bad header: {header} (expected id,label)")

        n = 0
        for row in r:
            if not row:
                continue
            if len(row) < 2:
                raise RuntimeError(f"Bad row: {row}")
            pid = (row[0] or "").strip()
            labs = (row[1] or "").strip()
            if not pid:
                raise RuntimeError("Empty id")
            labels = [x.strip() for x in labs.split(",") if x.strip()]
            if len(labels) not in (2, 3):
                raise RuntimeError(f"pid={pid} has {len(labels)} labels (must be 2 or 3)")
            for lab in labels:
                if lab not in allowed:
                    raise RuntimeError(f"pid={pid} has invalid label: {lab}")
            n += 1

    if n != len(d.test_docs):
        raise RuntimeError(f"Row count mismatch: submission={n}, expected={len(d.test_docs)}")

    logging.info("Submission OK: %s", sub_path)


if __name__ == "__main__":
    main()
