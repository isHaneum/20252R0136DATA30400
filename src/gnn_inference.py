"""
GNN Classifier Inference Script

Perform inference on test data using trained GNN model.

Usage:
    python src/gnn_inference.py --model-dir student_gnn --output output/final_submission.csv

Colab Usage:
    python src/gnn_inference.py --model-dir /content/drive/MyDrive/project_llm/student_gnn \
        --data-dir /content/drive/MyDrive/project_llm/Amazon_products \
        --output /content/drive/MyDrive/project_llm/output/final_submission.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer
from tqdm.auto import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import Paths, load_config
from data import parse_all
from gnn_classifier import GNNMultiLabelClassifier
from utils import ensure_dir

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)


def _is_trained_model_dir(path: str) -> bool:
    if not path:
        return False
    return (
        os.path.isfile(os.path.join(path, "label2id.json"))
        and os.path.isfile(os.path.join(path, "model_state.pt"))
    )


def _resolve_model_dir(model_dir: str, project_root: str) -> str:
    """Resolve model_dir robustly across local/Colab and relative paths."""
    model_dir = model_dir or ""
    candidates: List[str] = []

    # 1) As provided (absolute or relative to CWD)
    candidates.append(model_dir)

    # 2) Relative to project root
    if not os.path.isabs(model_dir):
        candidates.append(os.path.join(project_root, model_dir))

    # 3) Common layout: artifacts/<model_dir>
    candidates.append(os.path.join(project_root, "artifacts", os.path.basename(model_dir)))

    # 4) If user passed artifacts/... already, also try project_root/artifacts/...
    if model_dir.replace("\\", "/").startswith("artifacts/"):
        candidates.append(os.path.join(project_root, model_dir))

    candidates = [os.path.abspath(p) for p in candidates if p]
    seen = set()
    candidates = [p for p in candidates if not (p in seen or seen.add(p))]

    for cand in candidates:
        if _is_trained_model_dir(cand):
            return cand

    # Auto-discovery: search a few levels under project root
    found: List[str] = []
    max_depth = 4
    root_depth = os.path.abspath(project_root).rstrip(os.sep).count(os.sep)
    for dirpath, dirnames, filenames in os.walk(project_root):
        cur_depth = os.path.abspath(dirpath).rstrip(os.sep).count(os.sep) - root_depth
        if cur_depth > max_depth:
            dirnames[:] = []
            continue
        if "label2id.json" in filenames and "model_state.pt" in filenames:
            found.append(dirpath)
            if len(found) >= 10:
                break

    details = "\n".join(f"- tried: {p}" for p in candidates)
    found_details = "\n".join(f"- found: {p}" for p in found) if found else "- (none found)"
    raise FileNotFoundError(
        "Could not locate a trained model directory. Expected files: label2id.json and model_state.pt\n\n"
        f"Attempted paths:\n{details}\n\n"
        f"Auto-discovery under project root ({project_root}):\n{found_details}\n\n"
        "Fix: pass --model-dir pointing to the folder created by training (--save-dir).\n"
        "Example (Colab): --model-dir /content/drive/MyDrive/project_llm/student_gnn"
    )


"""NOTE
We intentionally import `GNNMultiLabelClassifier` from `gnn_classifier.py`.
Do NOT duplicate model definitions here; otherwise state_dict keys/shapes diverge.
"""


# ============================================================================
# Inference Dataset
# ============================================================================

class TestDataset(Dataset):
    """Test dataset for inference"""
    
    def __init__(self, test_data: List[Tuple[int, str]]):
        self.data = test_data  # [(doc_id, text), ...]
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        doc_id, text = self.data[idx]
        return {"doc_id": doc_id, "text": text}


def make_test_collate(tokenizer, max_length: int):
    def collate_fn(batch):
        texts = [item["text"] for item in batch]
        doc_ids = [item["doc_id"] for item in batch]
        
        encoded = tokenizer(
            texts,
            max_length=max_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        
        return {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
            "doc_ids": doc_ids,
        }
    
    return collate_fn


# ============================================================================
# Main Inference Function
# ============================================================================

def load_model(model_dir: str, device: torch.device):
    """Load trained model"""

    # Resolve model directory robustly
    cfg = load_config()
    model_dir = _resolve_model_dir(model_dir, cfg.paths.project_root)
    logging.info(f"Resolved model dir: {model_dir}")
    
    # Load label2id
    with open(os.path.join(model_dir, "label2id.json"), "r") as f:
        label2id = json.load(f)
    id2label = {v: k for k, v in label2id.items()}
    
    num_labels = len(label2id)
    logging.info(f"Loaded {num_labels} labels")
    
    # Load edge index
    edge_index_path = os.path.join(model_dir, "edge_index.pt")
    if os.path.exists(edge_index_path):
        edge_index = torch.load(edge_index_path).to(device)
        logging.info(f"Loaded edge index with {edge_index.shape[1]} edges")
    else:
        edge_index = None
        logging.warning("No edge index found, running without GNN")
    
    # Create model (must match training architecture exactly)
    model = GNNMultiLabelClassifier(
        encoder_name=model_dir,  # saved encoder folder
        num_labels=num_labels,
        edge_index=edge_index,
        gnn_hidden_dim=256,
        use_gnn=True,
    )
    
    # Load state dict
    state_dict_path = os.path.join(model_dir, "model_state.pt")
    if os.path.exists(state_dict_path):
        state_dict = torch.load(state_dict_path, map_location=device)
        model.load_state_dict(state_dict, strict=True)
        logging.info("Loaded model state dict")
    
    model.to(device)
    model.eval()
    
    return model, label2id, id2label


def predict(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    id2label: Dict[int, str],
    child2parents: Dict[str, List[str]],
    parent2children: Dict[str, List[str]],
    max_len: int = 4,
) -> List[Tuple[int, List[str]]]:
    """
    Perform inference and return predictions
    
    Returns:
        List of (doc_id, [label1, label2, ...]) where len(labels) is 2 or 3.
    """

    def _select_labels_from_probs(prob_vec: torch.Tensor) -> List[str]:
        """Selection rule (matches baseline inference.py):

        - Anchor on top-1 predicted label (primary)
                - Output the taxonomy chain in bottom-up order: [primary, parent, grandparent] (up to 3)
                - If any *leaf* label has prob >= 0.4, add exactly ONE extra label:
                    the highest-prob leaf not already included.
                - Always return between 2 and `max_len` labels
        """
        max_labels = 3  # Ensure max_labels is 3 by default
        if max_labels < 2:
            max_labels = 2
        order = torch.argsort(prob_vec, descending=True)
        primary = str(id2label[int(order[0].item())])

        # Build taxonomy chain up to 3 (primary->parent->grandparent)
        path: List[str] = [primary]
        cur = primary
        while len(path) < min(3, max_labels):
            parents = child2parents.get(cur) or []
            if not parents:
                break
            cur = str(parents[0])
            if cur in path:
                break
            path.append(cur)
        chosen = list(path)

        # Ensure at least 2 labels and at most 3 labels initially
        if len(chosen) < 2:
            chosen.append(primary)
        chosen = chosen[:3]  # Limit to 3 labels initially

        # Add ONE extra label if prob >= 0.6 and we have room (<4 labels total)
        if len(chosen) < 4:
            included = set(chosen)
            for idx in order[1:]:
                cid = str(id2label[int(idx.item())])
                if cid in included:
                    continue
                # Check if the candidate is a leaf node
                if cid in parent2children and len(parent2children.get(cid, [])) > 0:
                    continue
                if float(prob_vec[int(idx.item())].item()) >= 0.5:  # Update threshold to 0.5 for extra leaf selection
                    chosen.append(cid)
                    break

        if len(chosen) >= max_labels:
            return chosen[:max_labels]
        if len(chosen) == 2:
            return chosen
        if len(chosen) == 1:
            return [chosen[0], chosen[0]]
        return ["0", "1"]
    
    model.eval()
    predictions = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Inference"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            doc_ids = batch["doc_ids"]

            use_amp = device.type == "cuda" and hasattr(torch, "amp") and hasattr(torch.amp, "autocast")
            autocast_ctx = torch.amp.autocast("cuda", enabled=use_amp) if use_amp else torch.no_grad()
            with autocast_ctx:
                logits = model(input_ids, attention_mask)

            probs = torch.sigmoid(logits)

            for i, doc_id in enumerate(doc_ids):
                labels = _select_labels_from_probs(probs[i])
                # Enforce 2..max_len.
                if len(labels) < 2:
                    labels = labels + labels[:1]
                labels = labels[: int(max_len) if int(max_len) >= 2 else 2]
                if len(labels) == 1:
                    labels = labels + labels
                predictions.append((doc_id, labels))
    
    return predictions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=str, default="student_gnn",
                        help="Directory containing the trained model")
    parser.add_argument("--data-dir", type=str, default=None,
                        help="Directory containing Amazon_products (default: from config)")
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output CSV file path (default: config submission_file)",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--top-k", type=int, default=4,
                        help="Max number of labels to output per document (2, 3, or 4)")
    parser.add_argument("--student-id", type=str, default="2021320045",
                        help="Student ID for submission filename")
    args = parser.parse_args()

    if args.top_k not in (2, 3, 4):
        raise SystemExit("--top-k must be 2, 3, or 4")
    
    # Load config
    cfg = load_config()
    
    # Data directory (robust: allow passing either Amazon_products/ or project root)
    data_dir = os.path.abspath(args.data_dir or cfg.paths.data_dir)
    if os.path.isdir(data_dir):
        maybe_classes = os.path.join(data_dir, "classes.txt")
        if not os.path.exists(maybe_classes):
            maybe_data_dir = os.path.join(data_dir, "Amazon_products")
            if os.path.exists(os.path.join(maybe_data_dir, "classes.txt")):
                data_dir = maybe_data_dir

    logging.info(f"Loading test data from {data_dir}")

    # Build a proper Paths object for data.py
    paths = Paths(project_root=cfg.paths.project_root, data_dir=data_dir)

    # Validate required inputs early (prevents confusing AttributeError later)
    required_files = [
        ("classes.txt", paths.classes_file),
        ("class_hierarchy.txt", paths.hierarchy_file),
        ("class_related_keywords.txt", paths.keyword_file),
        ("train/train_corpus.txt", paths.train_corpus),
        ("test/test_corpus.txt", paths.test_corpus),
    ]
    missing = [f"{name} -> {path}" for name, path in required_files if not os.path.exists(path)]
    if missing:
        raise SystemExit(
            "Data directory is not valid. Missing required files:\n"
            + "\n".join(missing)
            + "\n\nTip: pass --data-dir pointing to the Amazon_products folder (or the project root containing it)."
        )

    # Parse dataset
    d = parse_all(paths)
    
    # Test data: [(doc_id, text), ...]
    test_data = [(doc_id, text) for doc_id, text in d.test_docs.items()]
    logging.info(f"Loaded {len(test_data)} test documents")
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Device: {device}")
    
    # Load model
    logging.info(f"Loading model from {args.model_dir}")
    model, label2id, id2label = load_model(args.model_dir, device)
    
    # Tokenizer
    # Tokenizer must load from the resolved model dir (same folder as encoder files)
    resolved_model_dir = _resolve_model_dir(args.model_dir, cfg.paths.project_root)
    tokenizer = AutoTokenizer.from_pretrained(resolved_model_dir, use_fast=False)
    
    # Dataset and DataLoader
    test_ds = TestDataset(test_data)
    test_dl = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=make_test_collate(tokenizer, args.max_length),
    )
    
    # Inference
    logging.info("Running inference...")
    predictions = predict(
        model,
        test_dl,
        device,
        id2label,
        child2parents=d.child2parents,
        parent2children=d.parent2children,
        max_len=int(args.top_k),
    )
    
    # Sort by doc_id
    predictions.sort(key=lambda x: x[0])
    
    out_path = str(args.output) if args.output else str(cfg.paths.submission_file)
    ensure_dir(os.path.dirname(out_path))

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "label"])
        for doc_id, labels in predictions:
            w.writerow([doc_id, ",".join(labels)])

    logging.info("Saved submission to %s", out_path)
    
    # Summary
    logging.info("\n" + "="*50)
    logging.info("INFERENCE COMPLETE")
    logging.info(f"  - Total predictions: {len(predictions)}")
    logging.info(f"  - Labels per doc: {int(args.top_k)}")
    logging.info(f"  - Output: {out_path}")
    logging.info("="*50)


if __name__ == "__main__":
    main()
