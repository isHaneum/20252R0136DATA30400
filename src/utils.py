from __future__ import annotations

import os
import json
import random
import warnings
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List

import numpy as np
import torch
import matplotlib.pyplot as plt
from collections import defaultdict
import itertools

# Suppress warnings
warnings.filterwarnings("ignore")

# Suppress HuggingFace transformers logs
from transformers import logging
logging.set_verbosity_error()


def ensure_dir(path: str) -> None:
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def write_text(path: str, text: str) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        return []
    out: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            raw = raw.strip()
            if not raw:
                continue
            try:
                out.append(json.loads(raw))
            except Exception:
                continue
    return out


def write_jsonl(path: str, rows: Iterable[Dict[str, Any]]) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def append_jsonl(path: str, rows: Iterable[Dict[str, Any]]) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def plot_results(results_dict, split="valid", metric='Accuracy'):
    """
    Plot metric values over epochs for multiple models.

    Args:
        results_dict: dict of dicts.
            Example: results_dict["valid"]["mlp_partial"] = [0.8, 0.82, ...]
        split: "valid" or "test"
        metric: name of the metric to display
    """
    assert split in results_dict, f"{split} not in results_dict"

    plt.figure(figsize=(8, 5))
    for label, acc_list in results_dict[split].items():
        plt.plot(acc_list, label=label)

    plt.title(f"{split.capitalize()} {metric} over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel(metric)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def print_eval_result(metrics: dict, stage="val", is_improved=False):
    """
    Print evaluation results (accuracy, F1-macro).

    Args:
        metrics: dict with keys 'accuracy' and 'f1_macro'
        stage: string label (e.g., "val", "test")
        is_improved: mark with '*' if results improved
    """
    star = " *" if is_improved else ""
    print(f"[{stage.upper():4}] Acc: {metrics['accuracy']:.4f} | F1-macro: {metrics['f1_macro']:.4f}{star}")


def build_leaf_adj(id2label, label2id, decay=0.01, max_edges=10):
    """
    Build an adjacency matrix between leaf nodes based on hierarchical structure.

    Args:
        id2label: {id(str): full_path_str}, mapping from label ID to its full path
        label2id: {full_path_str: id(str)}, mapping from path string to ID
        decay: decay factor for ancestor depth (shallower common ancestors → lower weight)
        max_edges: maximum number of edges per node (keep top-k by weight)

    Returns:
        A_hat (torch.FloatTensor): normalized adjacency matrix (D^{-1/2} A D^{-1/2})
    """
    leaf2ancestors = {}
    for leaf_id, full_path in id2label.items():
        parts = full_path.split(" > ")
        ancestors = [" > ".join(parts[:d]) for d in range(1, len(parts))]
        leaf2ancestors[leaf_id] = ancestors

    n_labels = len(label2id)
    A = np.eye(n_labels, dtype=np.float32)

    all_ids = list(id2label.keys())

    edges_by_node = defaultdict(list)
    for u, v in itertools.combinations(all_ids, 2):
        anc_u, anc_v = set(leaf2ancestors[u]), set(leaf2ancestors[v])
        common = anc_u.intersection(anc_v)
        if not common:
            continue

        max_depth = max(len(c.split(" > ")) for c in common)
        depth_u = len(id2label[u].split(" > "))
        weight = decay ** (depth_u - max_depth)

        iu, iv = int(u), int(v)
        edges_by_node[iu].append((iv, weight))
        edges_by_node[iv].append((iu, weight))

    for u, neigh_list in edges_by_node.items():
        top_neighbors = sorted(neigh_list, key=lambda x: x[1], reverse=True)[:max_edges]
        for v, w in top_neighbors:
            A[u, v] = max(A[u, v], w)
            A[v, u] = max(A[v, u], w)

    D = np.sum(A, axis=1)
    D_inv_sqrt = np.diag(1.0 / np.sqrt(D + 1e-8))
    A_hat = D_inv_sqrt @ A @ D_inv_sqrt

    return torch.from_numpy(A_hat).float()
