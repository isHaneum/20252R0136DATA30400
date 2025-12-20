from __future__ import annotations

import argparse
import json
import logging
import os
from typing import Dict, List, Tuple

import torch
import numpy as np
from config import load_config, TAXONOMY_PATH
from data import parse_all
from utils import ensure_dir


def build_graph(child2parents: Dict[str, List[str]]) -> Tuple[List[Tuple[str, str]], List[str]]:
    nodes = set(child2parents.keys())
    edges: List[Tuple[str, str]] = []
    for child, parents in child2parents.items():
        for parent in parents:
            nodes.add(parent)
            edges.append((parent, child))
    return edges, sorted(nodes)


def build_adjacency_matrix(num_classes=None):
    print(f"Building Taxonomy Graph from {TAXONOMY_PATH}")

    edges = []
    max_id = 0

    if os.path.exists(TAXONOMY_PATH):
        with open(TAXONOMY_PATH, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    try:
                        p, c = int(parts[0]), int(parts[1])
                        edges.append((p, c))
                        max_id = max(max_id, p, c)
                    except ValueError:
                        continue
    else:
        print(f"Taxonomy file not found: {TAXONOMY_PATH}")
        return None

    if num_classes is None:
        num_classes = max_id + 1

    print(f"Graph Nodes: {num_classes}, Edges: {len(edges)}")

    adj = torch.eye(num_classes)

    for p, c in edges:
        if p < num_classes and c < num_classes:
            adj[p, c] = 1
            adj[c, p] = 1

    row_sum = torch.sum(adj, dim=1)
    d_inv_sqrt = torch.pow(row_sum, -0.5)
    d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = torch.diag(d_inv_sqrt)
    norm_adj = torch.mm(torch.mm(d_mat_inv_sqrt, adj), d_mat_inv_sqrt)

    return norm_adj


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-matrix", action="store_true", help="Build adjacency matrix instead of JSON graph")
    args = parser.parse_args()

    cfg = load_config()

    if args.build_matrix:
        adj = build_adjacency_matrix()
        if adj is not None:
            print(f"✅ Adjacency Matrix Created: {adj.shape}")
    else:
        d = parse_all(cfg.paths)
        edges, nodes = build_graph(d.child2parents)

        ensure_dir(cfg.paths.artifacts_dir)
        with open(cfg.paths.graph_file, "w", encoding="utf-8") as f:
            json.dump({"nodes": nodes, "edges": edges}, f, ensure_ascii=False)

        logging.info("Wrote graph: %s (nodes=%d edges=%d)", cfg.paths.graph_file, len(nodes), len(edges))


if __name__ == "__main__":
    main()
