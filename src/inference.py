<<<<<<< HEAD
from __future__ import annotations

import csv
import json
import logging
import os
import argparse
from typing import Dict, List, Set, Tuple

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from tqdm.auto import tqdm

from config import load_config
from data import parse_all
from utils import ensure_dir, read_jsonl



def _load_candidates(path: str) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    for r in read_jsonl(path):
        if not isinstance(r, dict) or "id" not in r:
            continue
        out[str(r["id"])] = [str(x) for x in r.get("candidates", [])]
    return out


def _expand_taxonomy_path(
    *,
    leaf_id: str,
    child2parents: Dict[str, List[str]],
    max_len: int = 3,
) -> List[str]:
    """Return up to `max_len` labels along the taxonomy path ending at leaf_id.

    Output order is parent -> ... -> child (top-down).
    """
    path = [str(leaf_id)]
    cur = str(leaf_id)
    while len(path) < int(max_len):
        parents = child2parents.get(cur) or []
        if not parents:
            break
        cur = str(parents[0])  # deterministic
        if cur in path:
            break
        path.append(cur)
    return list(reversed(path))


def _load_golden_labels(path: str) -> Dict[str, str]:
    """Load id,label CSV into a mapping. Values are raw comma-separated strings."""
    out: Dict[str, str] = {}
    if not path or not os.path.exists(path):
        return out
    with open(path, "r", encoding="utf-8") as f:
        header = f.readline()
        if not header:
            return out
        # accept either: id,label or pid,labels
        for line in f:
            line = line.strip()
            if not line:
                continue
            # naive CSV split (labels may be quoted) -> use csv module for safety
    with open(path, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            if not row:
                continue
            # tolerate both key styles
            doc_id = row.get("id") or row.get("pid")
            label = row.get("label") or row.get("labels")
            if doc_id is None or label is None:
                continue
            out[str(doc_id)] = str(label).strip().strip('"')
    return out


def _is_leaf(class_id: str, parent2children: Dict[str, List[str]]) -> bool:
    """Check if a class is a leaf node (has no children)."""
    return class_id not in parent2children or len(parent2children.get(class_id, [])) == 0


def _select_labels_dynamic(
    *,
    cand_ids: List[str],
    probs: np.ndarray,
    parent2children: Dict[str, List[str]],
    threshold: float,
    leaf_threshold: float,
    min_labels: int,
    max_labels: int,
    prefer_leaves: bool,
) -> List[str]:
    """Select labels dynamically based on threshold, with leaf preference.
    
    Strategy for F1 optimization:
    1. Include all labels with prob >= threshold
    2. For leaf nodes, use a lower threshold (leaf_threshold) since they're more specific
    3. Ensure at least min_labels, at most max_labels
    4. If prefer_leaves, sort by (is_leaf, prob) to prioritize leaves
    """
    if not cand_ids or len(probs) == 0:
        return ["0", "1"]
    
    pairs = list(zip(cand_ids, probs.tolist()))
    
    # Separate into above-threshold and below-threshold
    chosen: List[Tuple[str, float, bool]] = []
    below_threshold: List[Tuple[str, float, bool]] = []
    
    for cid, p in pairs:
        is_leaf = _is_leaf(cid, parent2children)
        effective_threshold = leaf_threshold if is_leaf else threshold
        
        if p >= effective_threshold:
            chosen.append((cid, p, is_leaf))
        else:
            below_threshold.append((cid, p, is_leaf))
    
    # Sort chosen: if prefer_leaves, put leaves first, then by prob desc
    if prefer_leaves:
        chosen.sort(key=lambda x: (-int(x[2]), -x[1]))
    else:
        chosen.sort(key=lambda x: -x[1])
    
    # Sort below_threshold similarly (for fallback)
    if prefer_leaves:
        below_threshold.sort(key=lambda x: (-int(x[2]), -x[1]))
    else:
        below_threshold.sort(key=lambda x: -x[1])
    
    # Ensure min_labels
    while len(chosen) < min_labels and below_threshold:
        chosen.append(below_threshold.pop(0))
    
    # Fallback if still not enough
    while len(chosen) < min_labels:
        if chosen:
            chosen.append(chosen[0])  # duplicate
        else:
            chosen.append(("0", 0.0, False))
    
    # Limit to max_labels
    chosen = chosen[:max_labels]
    
    return [cid for cid, _, _ in chosen]


def _select_labels(
    *,
    cand_ids: List[str],
    probs: np.ndarray,
    threshold: float,
    selection: str,
    third_ratio: float,
    third_margin: float,
) -> List[str]:
    pairs = list(zip(cand_ids, probs.tolist()))
    pairs.sort(key=lambda x: x[1], reverse=True)

    # Always ensure 2 labels exist if any candidates exist.
    if not pairs:
        return ["0", "1"]
    if len(pairs) == 1:
        return [pairs[0][0], pairs[0][0]]

    # Base selection
    selection = str(selection)
    thr = float(threshold)
    third_ratio = float(third_ratio)
    third_margin = float(third_margin)

    if selection == "threshold":
        chosen = [cid for cid, p in pairs if float(p) >= thr][:3]
        # fill up to 2
        if len(chosen) < 2:
            for cid, _ in pairs:
                if cid not in chosen:
                    chosen.append(cid)
                if len(chosen) >= 2:
                    break
        # relax 3rd: allow if it's close to #2 even if < threshold
        if len(chosen) == 2 and len(pairs) >= 3:
            p2 = float([p for cid, p in pairs if cid == chosen[1]][0])
            cid3, p3 = pairs[2][0], float(pairs[2][1])
            if cid3 not in chosen:
                if (p3 >= thr) or (p3 >= p2 * third_ratio) or ((p2 - p3) <= third_margin):
                    chosen.append(cid3)
        return chosen[:3]

    # selection == "topk" (default)
    chosen = [pairs[0][0], pairs[1][0]]
    if len(pairs) >= 3:
        p2 = float(pairs[1][1])
        cid3, p3 = pairs[2][0], float(pairs[2][1])
        if (p3 >= thr) or (p3 >= p2 * third_ratio) or ((p2 - p3) <= third_margin):
            chosen.append(cid3)
    return chosen[:3]


def _select_taxonomy_labels(
    *,
    cand_ids: List[str],
    probs: np.ndarray,
    child2parents: Dict[str, List[str]],
    parent2children: Dict[str, List[str]],
    threshold: float,
    selection: str,
    third_ratio: float,
    third_margin: float,
    max_len: int = 3,
) -> List[str]:
    """Select labels per user spec.

        Rule:
            - Anchor on top-1 predicted label (primary)
            - Output the taxonomy chain in bottom-up order: [primary, parent, grandparent] (up to 3)
            - If any *leaf* label has prob >= threshold, add exactly ONE extra label:
              the highest-prob leaf not already included.
            - Always output between 2 and max_len labels (by padding/duplicating if needed).
    """
    if not cand_ids or len(probs) == 0:
        return ["0", "1"]

    pairs = list(zip([str(x) for x in cand_ids], probs.tolist()))
    pairs.sort(key=lambda x: float(x[1]), reverse=True)

    max_labels = int(max_len)
    if max_labels < 2:
        max_labels = 2

    primary = str(pairs[0][0])

    # Build taxonomy path up to 3 (top-down), then flip to bottom-up:
    # [grandparent, parent, primary] -> [primary, parent, grandparent]
    path_top_down = _expand_taxonomy_path(
        leaf_id=primary,
        child2parents=child2parents,
        max_len=min(3, max_labels),
    )
    chosen: List[str] = [str(x) for x in reversed(path_top_down) if str(x)]

    # If we couldn't build 3-chain (missing parents), pad with primary.
    while len(chosen) < min(3, max_labels):
        chosen.append(primary)

    extra_thr = 0.6  # Update threshold to 0.6 for extra leaf selection

    # If any leaf has prob >= threshold, add exactly one best leaf.
    if len(chosen) < max_labels:
        included = set(chosen)
        for cid, p in pairs[1:]:
            cid = str(cid)
            if cid in included:
                continue
            if not _is_leaf(cid, parent2children):
                continue
            if float(p) >= extra_thr:
                chosen.append(cid)
                break

    # Enforce 2..max_labels labels.
    if len(chosen) >= max_labels:
        return chosen[:max_labels]
    if len(chosen) >= 2:
        return chosen
    if len(chosen) == 1:
        return [chosen[0], chosen[0]]
    return ["0", "1"]


def _load_label2id_from_model_dir(model_dir: str) -> Dict[str, int] | None:
    p = os.path.join(model_dir, "label2id.json")
    if not os.path.exists(p):
        return None
    try:
        with open(p, "r", encoding="utf-8") as f:
            obj = json.load(f)
    except Exception:
        return None
    if not isinstance(obj, dict):
        return None
    out: Dict[str, int] = {}
    for k, v in obj.items():
        try:
            out[str(k)] = int(v)
        except Exception:
            continue
    return out or None


def _label_order_from_label2id(label2id: Dict[str, int]) -> List[str]:
    return [cid for cid, _ in sorted(label2id.items(), key=lambda kv: kv[1])]


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    cfg = load_config()

    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=cfg.infer.batch_size)
    parser.add_argument("--selection", type=str, default=cfg.infer.selection, choices=["topk", "threshold", "dynamic"])
    parser.add_argument("--threshold", type=float, default=cfg.infer.threshold)
    parser.add_argument(
        "--third-ratio",
        type=float,
        default=cfg.infer.drop_ratio,
        help="Allow 3rd label if p3 >= p2 * third_ratio (lower => more 3-label outputs)",
    )
    parser.add_argument(
        "--third-margin",
        type=float,
        default=0.0,
        help="Allow 3rd label if (p2 - p3) <= third_margin",
    )
    # Dynamic mode options
    parser.add_argument("--min-labels", type=int, default=cfg.infer.min_labels)
    parser.add_argument("--max-labels", type=int, default=cfg.infer.max_labels)
    parser.add_argument("--leaf-threshold", type=float, default=cfg.infer.leaf_threshold)
    parser.add_argument("--prefer-leaves", action="store_true", default=cfg.infer.prefer_leaves)
    parser.add_argument("--max-docs", type=int, default=None)
    parser.add_argument("--use-llm", action="store_true", default=cfg.llm.enabled)
    parser.add_argument(
        "--sanity-silver",
        action="store_true",
        help="Run a quick sanity check by predicting on a subset of silver data and reporting agreement.",
    )
    parser.add_argument(
        "--golden-file",
        type=str,
        default=None,
        help="Optional CSV file (id,label) to override predictions for those ids (e.g., provided sample).",
    )
    args = parser.parse_args()

    model_dir = "student"
    if not os.path.exists(model_dir):
        model_dir = os.path.join(cfg.paths.model_dir, "student")
    
    if not os.path.exists(model_dir):
        raise RuntimeError(f"Model not found at {model_dir}. Run training first.")

    d = parse_all(cfg.paths)

    # Prefer the label mapping saved at training time to avoid silent index drift.
    label2id = _load_label2id_from_model_dir(model_dir)
    if label2id is None:
        class_ids = sorted(d.id2name.keys())
        label2id = {cid: i for i, cid in enumerate(class_ids)}
        logging.warning("label2id.json not found in %s; using sorted(id2name.keys()) mapping", model_dir)
    else:
        class_ids = _label_order_from_label2id(label2id)

        fresh_class_ids = sorted(d.id2name.keys())
        fresh_label2id = {cid: i for i, cid in enumerate(fresh_class_ids)}
        if fresh_label2id != label2id:
            # This is a common cause of 'everything is wrong' symptoms.
            overlap = len(set(fresh_label2id.keys()) & set(label2id.keys()))
            logging.warning(
                "label2id mismatch vs current dataset (overlap=%d/%d). Using model's label2id.json.",
                overlap,
                len(label2id),
            )

    candidates = _load_candidates(cfg.paths.candidates_test)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=False)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.to(device)
    model.eval()

    ensure_dir(cfg.paths.output_dir)

    test_ids = list(d.test_docs.keys())
    if args.max_docs is not None:
        test_ids = test_ids[: int(args.max_docs)]

    golden = _load_golden_labels(str(args.golden_file)) if args.golden_file else {}

    rows: List[Tuple[str, str]] = []
    llm_queue: List[object] = []
    llm_budget_docs = int(cfg.llm.infer_max_docs)

    api_key = None
    call_openai_batched = None
    get_openai_key = None
    LLMDoc = None
    if bool(args.use_llm) and (not bool(cfg.llm.use_in_inference)):
        logging.info("Inference LLM refinement is disabled (cfg.llm.use_in_inference=False); skipping API calls.")

    if bool(args.use_llm) and bool(cfg.llm.use_in_inference):
        # Lazy import so inference works without openai when LLM is disabled.
        from llm_utils import LLMDoc as _LLMDoc, call_openai_batched as _call_openai_batched, get_openai_key as _get_openai_key

        LLMDoc = _LLMDoc
        call_openai_batched = _call_openai_batched
        get_openai_key = _get_openai_key
        api_key = get_openai_key(key_file=cfg.paths.openai_key_file)
        if not api_key:
            logging.warning("LLM enabled for inference but no API key found; skipping LLM refinement.")

    batch_ids: List[str] = []
    batch_texts: List[str] = []

    def flush_batch() -> None:
        nonlocal batch_ids, batch_texts, rows
        if not batch_ids:
            return

        enc = tokenizer(
            batch_texts,
            truncation=True,
            padding=True,
            max_length=cfg.train.max_length,
            return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            logits = model(**enc).logits
            probs_all = torch.sigmoid(logits).detach().cpu().numpy()

        for doc_id, probs in zip(batch_ids, probs_all):
            if golden and str(doc_id) in golden:
                rows.append((doc_id, golden[str(doc_id)]))
                continue

            cand_ids = candidates.get(doc_id, [])
            cand_idx = [label2id[c] for c in cand_ids if c in label2id]
            if cand_idx:
                cand_probs = probs[cand_idx]
                cand_keep = [cand_ids[i] for i, c in enumerate(cand_ids) if c in label2id]
                
                # Use dynamic selection mode for F1 optimization
                if str(args.selection) == "dynamic":
                    pred = _select_labels_dynamic(
                        cand_ids=cand_keep,
                        probs=cand_probs,
                        parent2children=d.parent2children,
                        threshold=float(args.threshold),
                        leaf_threshold=float(args.leaf_threshold),
                        min_labels=int(args.min_labels),
                        max_labels=int(args.max_labels),
                        prefer_leaves=bool(args.prefer_leaves),
                    )
                else:
                    pred = _select_taxonomy_labels(
                        cand_ids=cand_keep,
                        probs=cand_probs,
                        child2parents=d.child2parents,
                        parent2children=d.parent2children,
                        threshold=float(args.threshold),
                        selection=str(args.selection),
                        third_ratio=float(args.third_ratio),
                        third_margin=float(args.third_margin),
                        max_len=int(args.max_labels),
                    )

                # LLM refinement candidate selection
                if api_key and len(llm_queue) < llm_budget_docs:
                    order = np.argsort(-cand_probs)
                    sorted_scores = cand_probs[order]
                    sorted_ids = [cand_keep[i] for i in order]
                    if cfg.llm.infer_policy == "always":
                        need_llm = True
                    else:
                        if len(sorted_scores) >= 3:
                            margin23 = float(sorted_scores[1]) - float(sorted_scores[2])
                        else:
                            margin23 = 1.0
                        need_llm = float(sorted_scores[1]) < float(cfg.llm.low_conf) or margin23 < float(cfg.llm.small_margin)

                    if need_llm:
                        topn = min(int(cfg.llm.infer_top_n), len(sorted_ids))
                        cand_pairs = [(cid, d.id2name.get(cid, "")) for cid in sorted_ids[:topn]]
                        llm_queue.append(LLMDoc(doc_id=doc_id, text=d.test_docs[doc_id], candidates=cand_pairs))
            else:
                # fallback: global top (no candidates)
                top = np.argsort(-probs)
                if str(args.selection) == "dynamic":
                    # Dynamic fallback: use threshold on all classes
                    all_cids = [class_ids[i] for i in top]
                    all_probs = probs[top]
                    pred = _select_labels_dynamic(
                        cand_ids=all_cids,
                        probs=all_probs,
                        parent2children=d.parent2children,
                        threshold=float(args.threshold),
                        leaf_threshold=float(args.leaf_threshold),
                        min_labels=int(args.min_labels),
                        max_labels=int(args.max_labels),
                        prefer_leaves=bool(args.prefer_leaves),
                    )
                else:
                    top1_leaf = class_ids[int(top[0])]
                    max_labels = int(args.max_labels)
                    if max_labels < 2:
                        max_labels = 2

                    path_top_down = _expand_taxonomy_path(
                        leaf_id=top1_leaf,
                        child2parents=d.child2parents,
                        max_len=min(3, max_labels),
                    )
                    pred = [str(x) for x in reversed(path_top_down) if str(x)]
                    while len(pred) < min(3, max_labels):
                        pred.append(str(top1_leaf))

                    # Add one best leaf above threshold if available.
                    if len(pred) < max_labels:
                        included = set(pred)
                        extra_thr = float(args.threshold)
                        for j in top[1:]:
                            cid = str(class_ids[int(j)])
                            if cid in included:
                                continue
                            if not _is_leaf(cid, d.parent2children):
                                continue
                            if float(probs[int(j)]) >= extra_thr:
                                pred.append(cid)
                                break

                    if len(pred) < 2:
                        pred = pred + pred[:1]

            rows.append((doc_id, ",".join(pred)))

        batch_ids, batch_texts = [], []

    for doc_id in tqdm(test_ids, desc="inference", unit="doc"):
        batch_ids.append(doc_id)
        batch_texts.append(d.test_docs[doc_id])
        if len(batch_ids) >= int(args.batch_size):
            flush_batch()

    flush_batch()

    # LLM refinement (batched) for selected ambiguous docs
    if api_key and llm_queue:
        logging.info("LLM refining %d docs (policy=%s)", len(llm_queue), cfg.llm.infer_policy)
        calls_dir = os.path.join(cfg.paths.llm_dir, "requests")
        refined_map: Dict[str, List[str]] = {}
        for i in range(0, len(llm_queue), int(cfg.llm.docs_per_call)):
            chunk = llm_queue[i : i + int(cfg.llm.docs_per_call)]
            refined = call_openai_batched(
                api_key=api_key,
                model=cfg.llm.model,
                temperature=float(cfg.llm.temperature),
                docs=chunk,
                calls_jsonl=cfg.paths.llm_calls_jsonl,
                calls_dir=calls_dir,
                max_calls_total=int(cfg.llm.max_calls),
                rpm=int(cfg.llm.max_rpm),
            )
            refined_map.update(refined)

        if refined_map:
            refined_set = set(refined_map.keys())
            new_rows: List[Tuple[str, str]] = []
            for pid, labels in rows:
                if pid in refined_set:
                    new_rows.append((pid, ",".join(refined_map[pid])))
                else:
                    new_rows.append((pid, labels))
            rows = new_rows

    # Optional sanity check: compare predictions vs silver labels on a subset of training docs.
    if bool(args.sanity_silver):
        silver = [r for r in read_jsonl(cfg.paths.silver_file) if isinstance(r, dict) and "id" in r and "text" in r and "labels" in r]
        if not silver:
            logging.warning("sanity_silver requested but no rows found in %s", cfg.paths.silver_file)
        else:
            # sample first N for determinism
            n = min(2000, len(silver))
            gold = {str(r["id"]): [str(x) for x in r["labels"]] for r in silver[:n]}
            ids = list(gold.keys())
            texts = [str(silver[i]["text"]) for i in range(n)]

            preds: Dict[str, List[str]] = {}
            for i in range(0, n, int(args.batch_size)):
                b_ids = ids[i : i + int(args.batch_size)]
                b_texts = texts[i : i + int(args.batch_size)]
                enc = tokenizer(
                    b_texts,
                    truncation=True,
                    padding=True,
                    max_length=cfg.train.max_length,
                    return_tensors="pt",
                )
                enc = {k: v.to(device) for k, v in enc.items()}
                with torch.no_grad():
                    probs_all = torch.sigmoid(model(**enc).logits).detach().cpu().numpy()
                for doc_id, probs in zip(b_ids, probs_all):
                    top = np.argsort(-probs)[:3]
                    preds[doc_id] = [class_ids[j] for j in top]

            # Agreement: how often at least 1 of top3 hits a silver label
            hit1 = 0
            hit_any = 0
            for doc_id, gold_labels in gold.items():
                p = preds.get(doc_id, [])
                if p and p[0] in gold_labels:
                    hit1 += 1
                if any(x in gold_labels for x in p):
                    hit_any += 1
            logging.info(
                "sanity_silver@%d: top1_hit=%.3f top3_any_hit=%.3f",
                n,
                hit1 / max(1, n),
                hit_any / max(1, n),
            )

    # Always-on local smoke check on ~50 rows (fast, catches format drift).
    smoke_n = min(50, len(rows))
    if smoke_n > 0:
        allowed = set(d.id2name.keys())
        empty_id = 0
        bad_label_count = 0
        invalid_labels = 0
        for pid, labs in rows[:smoke_n]:
            pid = str(pid).strip()
            if not pid:
                empty_id += 1
                continue
            labels = [x.strip() for x in str(labs).split(",") if x.strip()]
            if len(labels) not in (2, 3):
                bad_label_count += 1
            for lab in labels:
                if lab not in allowed:
                    invalid_labels += 1
        logging.info(
            "smoke_submit@%d: empty_id=%d bad_label_count=%d invalid_labels=%d",
            smoke_n,
            empty_id,
            bad_label_count,
            invalid_labels,
        )

    with open(cfg.paths.submission_file, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        # Required schema: id,label (labels are comma-separated inside a single quoted cell)
        w.writerow(["id", "label"])
        for pid, labels in rows:
            w.writerow([pid, labels])

    logging.info("Wrote submission: %s (%d rows)", cfg.paths.submission_file, len(rows))


if __name__ == "__main__":
    main()
=======
import os
import torch
import pandas as pd
import json
from tqdm import tqdm
from torch.utils.data import DataLoader

import config
from models import GCNClassifier
from graph_build import build_adjacency_matrix
from training import GraphDataset # 데이터셋 클래스 재사용

def main():
    print(">>> Step 6: Inference & Submission...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. 설정 로드
    adj = build_adjacency_matrix().to(device)
    real_num_classes = adj.shape[0]
    
    # 2. 모델 로드
    model = GCNClassifier(doc_dim=768, label_dim=768, adj=adj, num_classes=real_num_classes).to(device)

    model_path = config.BEST_MODEL_PATH  # config에서 경로 가져오기
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print("Model weights loaded.")
    else:
        print("❌ Model weights not found. Please train first.")
        return

    # 3. 테스트 데이터 로드
    # 테스트용 데이터셋은 레이블이 없으므로 더미 레이블을 넣어 로드
    test_emb_path = os.path.join(config.EMB_DIR, "test_emb.pt")
    test_data = torch.load(test_emb_path)
    
    # 테스트 대상 ID 필터링 (존재하지 않을 경우 전체 테스트 임베딩의 PID 사용)
    test_pid_path = os.path.join(config.DATA_DIR, "category_classification", "pid2labelids_test.json")
    use_sequential_ids = False
    if os.path.exists(test_pid_path):
        with open(test_pid_path, 'r', encoding='utf-8') as f:
            target_pids = list(json.load(f).keys())
        output_ids = target_pids
        dummy_labels = {pid: [] for pid in target_pids}
    else:
        # 파일이 없으면 제출 형식(id=0..N-1)에 맞춰 순번 ID 사용
        use_sequential_ids = True
        output_ids = list(range(len(test_data['pids'])))
        # GraphDataset은 실제 PID로 인덱싱하므로 더미 라벨은 실제 PID로 작성
        dummy_labels = {pid: [] for pid in test_data['pids']}
    
    # 타겟 ID만 딕셔너리로 만듦 (GraphDataset 재활용을 위해)
    test_ds = GraphDataset(test_emb_path, dummy_labels, num_classes=real_num_classes)
    test_loader = DataLoader(test_ds, batch_size=config.BATCH_SIZE, shuffle=False)
    
    # 4. 예측
    model.eval()
    results = []
    
    print("Predicting...")
    with torch.no_grad():
        # GraphDataset은 (emb, label)을 반환하므로 label은 무시
        # 순서를 맞추기 위해 pid도 추적해야 함 -> test_ds.indices 순서대로 접근
        
        current_idx = 0
        for docs, _ in tqdm(test_loader):
            docs = docs.to(device)
            logits = model(docs)
            probs = torch.sigmoid(logits)
            
            # Threshold 0.5
            preds = (probs > 0.6).int().cpu().numpy()
            
            for pred_vec in preds:
                # 제출 형식에 맞춘 ID 선택
                pid = (output_ids[current_idx] if use_sequential_ids 
                       else test_ds.pids[test_ds.indices[current_idx]])
                
                # 1로 예측된 인덱스 추출
                indices = [str(i) for i, v in enumerate(pred_vec) if v == 1]
                
                # 하나도 예측 안 된 경우, Top 3
                if not indices:
                    top3 = torch.topk(probs[current_idx % preds.shape[0]], 3).indices.cpu().tolist()
                    indices = [str(i) for i in top3]
                
                # 라벨은 콤마로 구분: "3,21,56" 형태
                results.append({'id': pid, 'label': ",".join(indices)})
                current_idx += 1
                
    # 5. 저장
    submission_path = os.path.join(config.OUTPUT_DIR, "submission.csv")
    pd.DataFrame(results).to_csv(submission_path, index=False)
    print(f"🎉 Submission saved to {submission_path}")

if __name__ == "__main__":
    main()
>>>>>>> 8d078ba7d40ccf7a402f171edc4e82a60ef91e2d
