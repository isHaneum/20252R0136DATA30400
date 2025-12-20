from __future__ import annotations

import argparse
import csv
import logging
import os
from collections import Counter
from typing import Dict, List, Tuple

from config import load_config, Paths
from data import parse_all
from utils import ensure_dir


def _expand_to_parents(label: str, child2parents: Dict[str, List[str]], *, max_len: int) -> List[str]:
    """Deterministic parent expansion: label -> parent -> grandparent..."""
    out: List[str] = [label]
    cur = label
    while len(out) < max_len:
        parents = child2parents.get(cur) or []
        if not parents:
            break
        p = str(parents[0])
        if p in out:
            break
        out.append(p)
        cur = p
    return out


def _is_leaf(label: str, parent2children: Dict[str, List[str]]) -> bool:
    return not (label in parent2children and len(parent2children.get(label, [])) > 0)


def normalize_row(
    labels: List[str],
    *,
    k: int,
    child2parents: Dict[str, List[str]],
    parent2children: Dict[str, List[str]],
    fallback_label: str,
) -> List[str]:
    """Normalize arbitrary label lists to exactly k (2 or 3) labels.

    Policy (matches the report intent):
    - Anchor on a stable top-1 (first label in the row) and expand via taxonomy up to k.
    - If still short, add one extra leaf preferring any remaining leaf in the row.
    - If still short, fill with a deterministic fallback label (not duplication of primary unless unavoidable).
    - If too long, keep the taxonomy-expanded chain (max 3) and then optional extra leaf.

    Note: This cannot reproduce the original model probabilities (CSV has no scores),
    but keeps the "stable top-1 taxonomy" behavior and meets the 2~3 label rule.
    """

    labels = [str(x).strip() for x in labels if str(x).strip()]
    if k not in (2, 3):
        k = 3

    if not labels:
        labels = [fallback_label]

    primary = labels[0]
    chosen = _expand_to_parents(primary, child2parents, max_len=k)

    # Optional extra leaf (only if we still need more labels)
    if len(chosen) < k:
        included = set(chosen)
        extra: str | None = None

        # Prefer a remaining leaf from the original row
        for cand in labels[1:]:
            if cand in included:
                continue
            if _is_leaf(cand, parent2children):
                extra = cand
                break

        # Fallback: pick any remaining label from the original row
        if extra is None:
            for cand in labels[1:]:
                if cand not in included:
                    extra = cand
                    break

        if extra is not None:
            chosen.append(extra)

    # Final deterministic fill
    included = set(chosen)
    while len(chosen) < k:
        if fallback_label not in included:
            chosen.append(fallback_label)
            included.add(fallback_label)
        else:
            # As an absolute last resort, use primary.
            chosen.append(primary)

    return chosen[:k]


def postprocess(
    *,
    input_csv: str,
    output_csv: str,
    k: int,
    data_dir: str | None,
) -> Tuple[int, Counter]:
    cfg = load_config()

    # Data directory robustness
    root = cfg.paths.project_root
    resolved_data_dir = os.path.abspath(data_dir or cfg.paths.data_dir)
    if os.path.isdir(resolved_data_dir):
        if not os.path.exists(os.path.join(resolved_data_dir, "classes.txt")):
            maybe = os.path.join(resolved_data_dir, "Amazon_products")
            if os.path.exists(os.path.join(maybe, "classes.txt")):
                resolved_data_dir = maybe

    paths = Paths(project_root=root, data_dir=resolved_data_dir)
    d = parse_all(paths)

    # Most common label in the input becomes deterministic fallback.
    freq = Counter()
    with open(input_csv, "r", encoding="utf-8-sig", newline="") as f:
        r = csv.reader(f)
        _ = next(r, None)
        for row in r:
            if not row or len(row) < 2:
                continue
            labs = [x.strip() for x in (row[1] or "").split(",") if x.strip()]
            freq.update(labs)

    fallback_label = freq.most_common(1)[0][0] if freq else next(iter(d.id2name.keys()))

    ensure_dir(os.path.dirname(output_csv))
    hist = Counter()
    n = 0

    with open(input_csv, "r", encoding="utf-8-sig", newline="") as f_in, open(
        output_csv, "w", encoding="utf-8", newline=""
    ) as f_out:
        r = csv.reader(f_in)
        w = csv.writer(f_out)

        header = next(r, None)
        if not header or len(header) < 2:
            raise SystemExit("Bad input CSV header")

        w.writerow(["id", "label"])

        for row in r:
            if not row or len(row) < 2:
                continue
            pid = str(row[0]).strip()
            labs_raw = str(row[1] or "")
            labs = [x.strip() for x in labs_raw.split(",") if x.strip()]

            normalized = normalize_row(
                labs,
                k=k,
                child2parents=d.child2parents,
                parent2children=d.parent2children,
                fallback_label=str(fallback_label),
            )

            hist[len(normalized)] += 1
            w.writerow([pid, ",".join(normalized)])
            n += 1

    return n, hist


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    p = argparse.ArgumentParser(description="Normalize a submission CSV to exactly 2 or 3 labels per row")
    p.add_argument("--input", required=True, help="Input CSV path (id,label)")
    p.add_argument("--output", required=True, help="Output CSV path")
    p.add_argument("--k", type=int, default=3, help="2 or 3 labels per row")
    p.add_argument("--data-dir", type=str, default=None, help="Amazon_products dir (optional)")
    args = p.parse_args()

    if args.k not in (2, 3):
        raise SystemExit("--k must be 2 or 3")

    n, hist = postprocess(
        input_csv=os.path.abspath(args.input),
        output_csv=os.path.abspath(args.output),
        k=int(args.k),
        data_dir=args.data_dir,
    )

    logging.info("Wrote %d rows -> %s", n, os.path.abspath(args.output))
    logging.info("Labels-per-row histogram: %s", dict(hist))


if __name__ == "__main__":
    main()
