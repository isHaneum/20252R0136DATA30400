"""Inspect `data/Amazon_products` structure and print concise summaries.

Usage:
    python src/check_amazon_products.py

The script prints:
 - top-level folders and file counts
 - sizes for files
 - quick previews for important files (`corpus.jsonl`, `classes.txt`, `class_hierarchy.txt`, `label2labelid.json`)
"""
import os
import json
import sys
from pathlib import Path
from typing import Optional


DEFAULT_DATA_DIR = os.path.join(os.getcwd(), "data", "Amazon_products")


def human_size(nbytes: int) -> str:
    for unit in ['B','KB','MB','GB']:
        if nbytes < 1024.0:
            return f"{nbytes:.1f}{unit}"
        nbytes /= 1024.0
    return f"{nbytes:.1f}TB"


def list_tree(root: Path, max_depth: int = 2) -> None:
    print(f"\nDirectory tree for: {root}\n")
    if not root.exists():
        print(f"Path does not exist: {root}")
        return

    root = root.resolve()
    for dirpath, dirnames, filenames in os.walk(root):
        depth = Path(dirpath).relative_to(root).parts
        if len(depth) > max_depth:
            # skip deep recursion
            continue
        indent = '  ' * len(depth)
        print(f"{indent}{os.path.basename(dirpath)}/")
        for fname in sorted(filenames):
            fpath = Path(dirpath) / fname
            try:
                size = human_size(fpath.stat().st_size)
            except Exception:
                size = 'n/a'
            print(f"{indent}  - {fname} ({size})")


def preview_file(path: Path, nlines: int = 5) -> None:
    print(f"\n--- Preview: {path} ---")
    if not path.exists():
        print(f"(missing) {path}")
        return

    lower = path.suffix.lower()
    try:
        if lower in ['.json', '.jsonl'] or 'json' in path.name:
            # Show first n JSON lines or first JSON object
            with path.open('r', encoding='utf-8') as f:
                if lower == '.json':
                    data = json.load(f)
                    if isinstance(data, dict):
                        keys = list(data.keys())[:20]
                        print(f"Type: JSON object (keys[:20]) -> {keys}")
                    elif isinstance(data, list):
                        print(f"Type: JSON list, length={len(data)}; item0 keys={list(data[0].keys()) if data else None}")
                else:
                    # jsonl
                    for i, line in enumerate(f):
                        if i >= nlines:
                            break
                        try:
                            obj = json.loads(line)
                            # print top-level keys and sample values lengths
                            keys = list(obj.keys())
                            print(f"line {i+1}: keys={keys}")
                        except Exception:
                            print(f"line {i+1}: (not json)")
        else:
            with path.open('r', encoding='utf-8', errors='replace') as f:
                for i in range(nlines):
                    line = f.readline()
                    if not line:
                        break
                    print(f"{i+1}: {line.rstrip()}")
    except Exception as e:
        print(f"Error previewing file {path}: {e}")


def summarize_labelmap(path: Path) -> None:
    print(f"\n--- Label map summary: {path} ---")
    if not path.exists():
        print("(missing)")
        return
    try:
        with path.open('r', encoding='utf-8') as f:
            d = json.load(f)
        print(f"Entries: {len(d)}; sample 10 items:")
        for i, (k,v) in enumerate(d.items()):
            if i >= 10:
                break
            print(f"  {k} -> {v}")
    except Exception as e:
        print(f"Failed to read JSON: {e}")


def inspect(data_dir: Optional[str] = None) -> None:
    data_dir = data_dir or DEFAULT_DATA_DIR
    root = Path(data_dir)

    print("Amazon_products inspection")
    print("==========================")

    # 1) Tree
    list_tree(root, max_depth=3)

    # 2) Useful files to preview
    candidates = [
        root / 'corpus.jsonl',
        root / 'corpus' / 'corpus.jsonl',
        root / 'train' / 'train_corpus.txt',
        root / 'test' / 'test_corpus.txt',
        root / 'classes.txt',
        root / 'class_hierarchy.txt',
        root / 'class_related_keywords.txt',
        root / 'category_classification' / 'label2labelid.json',
        root / 'product_categories.json',
    ]

    for p in candidates:
        preview_file(p)

    # 3) label map summary (if exists at that path)
    labelmap = root / 'category_classification' / 'label2labelid.json'
    if labelmap.exists():
        summarize_labelmap(labelmap)


def main(argv):
    data_dir = None
    if len(argv) > 1:
        data_dir = argv[1]
    inspect(data_dir)


if __name__ == '__main__':
    main(sys.argv)
