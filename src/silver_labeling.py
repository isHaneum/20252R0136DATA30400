from __future__ import annotations

# 역할 role: silver labels
# 순서 order: after retrieval
# 왜 why: train targets

import argparse
import json
import logging
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from sentence_transformers import CrossEncoder, SentenceTransformer
from tqdm.auto import tqdm

from config import load_config
from utils import ensure_dir, read_jsonl, write_jsonl
from llm_utils import LLMDoc, call_openai_batched, get_openai_key


@dataclass
class SilverExample:
    id: str
    text: str
    labels: List[str]


def _load_candidates(path: str) -> Dict[str, List[str]]:
    data: Dict[str, List[str]] = {}
    for row in read_jsonl(path):
        doc_id = str(row["id"])
        cands = list(row.get("candidates", []))
        data[doc_id] = cands
    return data


def _build_class_texts(id2name: Dict[str, str], keywords: Dict[str, List[str]]) -> Tuple[List[str], List[str]]:
    class_ids = sorted(id2name.keys())
    texts: List[str] = []
    for cid in class_ids:
        name = id2name.get(cid, "")
        kw = keywords.get(cid, [])
        texts.append(" ".join([name] + kw))
    return class_ids, texts


def _pick_2_or_3(sorted_ids: List[str], sorted_scores: List[float], margin_for_third: float) -> Tuple[List[str], float]:
    # Expect sorted descending
    if not sorted_ids:
        return ["0", "1"], 0.0
    if len(sorted_ids) == 1:
        return [sorted_ids[0], sorted_ids[0]], float(sorted_scores[0])

    chosen = [sorted_ids[0], sorted_ids[1]]
    conf = float(sorted_scores[1])
    if len(sorted_ids) >= 3:
        if float(sorted_scores[2]) >= float(sorted_scores[1]) - margin_for_third:
            chosen.append(sorted_ids[2])
            conf = float(sorted_scores[2])
    return chosen, conf


# Global hierarchy lookup (will be set by generate_silver)
_CHILD2PARENTS: Dict[str, List[str]] = {}
_PARENT2CHILDREN: Dict[str, List[str]] = {}
_LEAVES: set = set()


def _init_hierarchy(child2parents: Dict[str, List[str]], all_ids: set) -> None:
    """Initialize global hierarchy lookup tables."""
    global _CHILD2PARENTS, _PARENT2CHILDREN, _LEAVES
    _CHILD2PARENTS = child2parents
    _PARENT2CHILDREN = {}
    for c, ps in child2parents.items():
        for p in ps:
            _PARENT2CHILDREN.setdefault(p, []).append(c)
    _LEAVES = all_ids - set(_PARENT2CHILDREN.keys())


def _expand_to_hierarchy_path(leaf_id: str, max_len: int = 3) -> List[str]:
    """Expand a leaf node to its hierarchy path (root -> ... -> leaf)."""
    path = [str(leaf_id)]
    cur = str(leaf_id)
    while len(path) < max_len:
        parents = _CHILD2PARENTS.get(cur, [])
        if not parents:
            break
        cur = str(parents[0])
        if cur in path:
            break
        path.append(cur)
    return list(reversed(path))  # root -> ... -> leaf order


def _pick_hierarchy_path(
    sorted_ids: List[str],
    sorted_scores: List[float],
    margin_for_third: float,
) -> Tuple[List[str], float]:
    """
    Pick the best leaf and expand to hierarchy path.
    
    Strategy:
    1. Find the highest-scoring LEAF node among candidates
    2. Expand it to a hierarchy path (root -> mid -> leaf)
    3. Return 2-3 labels forming a valid taxonomy path
    """
    if not sorted_ids:
        return ["0", "1"], 0.0
    
    # Find the best leaf node (highest score among leaves)
    best_leaf = None
    best_leaf_score = -1.0
    best_leaf_idx = -1
    
    for i, (cid, score) in enumerate(zip(sorted_ids, sorted_scores)):
        if str(cid) in _LEAVES:
            if score > best_leaf_score:
                best_leaf = str(cid)
                best_leaf_score = score
                best_leaf_idx = i
            break  # Take the first (highest scoring) leaf
    
    # Fallback: if no leaf found, use the top candidate
    if best_leaf is None:
        best_leaf = str(sorted_ids[0])
        best_leaf_score = float(sorted_scores[0])
        best_leaf_idx = 0
    
    # Expand to hierarchy path
    path = _expand_to_hierarchy_path(best_leaf, max_len=3)
    
    # Ensure at least 2 labels
    if len(path) < 2:
        # Add the next best non-overlapping candidate
        for cid in sorted_ids:
            if str(cid) not in path:
                path.append(str(cid))
                break
    
    # Decide 2 or 3 labels based on margin
    if len(path) >= 3:
        # Check if we should include the third label
        # We use the original margin logic but on the path
        chosen = path[:3]
        conf = best_leaf_score
    elif len(path) == 2:
        chosen = path[:2]
        conf = best_leaf_score
    else:
        chosen = path + [path[-1]]  # duplicate if only 1
        conf = best_leaf_score
    
    return chosen, conf


def _make_prompt(text: str, candidate_labels: List[Tuple[str, str]], max_labels: int = 3) -> str:
    lines = [
        "You are labeling an Amazon product description into 2 or 3 category IDs.",
        "Choose only from the candidate list.",
        "Return ONLY a JSON object with a single key 'labels' whose value is a list of 2 or 3 category IDs.",
        "Do not include any extra keys or text.",
        "",
        "CANDIDATES:",
    ]
    for cid, name in candidate_labels:
        lines.append(f"- {cid}: {name}")
    lines += [
        "",
        "TEXT:",
        text.strip(),
        "",
        f"Remember: output 2 or 3 labels only (max {max_labels}).",
    ]
    return "\n".join(lines)


def _parse_labels(s: str, allowed: set[str]) -> Optional[List[str]]:
    try:
        obj = json.loads(s)
    except Exception:
        return None
    if not isinstance(obj, dict) or "labels" not in obj:
        return None
    labels = obj["labels"]
    if not isinstance(labels, list):
        return None
    out: List[str] = []
    for x in labels:
        if not isinstance(x, str):
            return None
        x = x.strip()
        if x in allowed and x not in out:
            out.append(x)
    if len(out) not in (2, 3):
        return None
    return out


def _call_openai_chat(*, api_key: str, model: str, prompt: str, timeout_s: int) -> str:
    client = OpenAI(api_key=api_key, timeout=timeout_s)
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a careful classifier."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,
    )
    return resp.choices[0].message.content or ""


def generate_silver(
    *,
    train_docs: Dict[str, str],
    id2name: Dict[str, str],
    candidates_jsonl: str,
    out_jsonl: str,
    llm_log_dir: str,
    api_key_file: str,
    model: str,
    max_calls: int,
    rpm: int,
    max_docs: Optional[int] = None,
    force_no_llm: bool = False,
    # Open-model silver
    embed_model: str = "BAAI/bge-m3",
    embed_batch_size: int = 128,
    embed_trust_remote_code: bool = False,
    use_reranker: bool = True,
    reranker_model: str = "BAAI/bge-reranker-v2-m3",
    reranker_trust_remote_code: bool = False,
    rerank_top_n: int = 15,  # Reduced from 60 to 15 for speed (top candidates are usually correct)
    rerank_policy: str = "uncertain",  # Changed from "always" to "uncertain" for speed
    low_conf: float = 0.25,
    small_margin: float = 0.03,
    margin_for_third: float = 0.02,

    # LLM refinement (final optimization)
    llm_enabled: bool = False,
    llm_temperature: float = 0.0,
    llm_docs_per_call: int = 10,
    llm_max_docs: int = 3000,
    llm_policy: str = "uncertain",  # uncertain | always
    llm_top_n: int = 40,
) -> None:
    ensure_dir(os.path.dirname(out_jsonl))
    ensure_dir(llm_log_dir)

    api_key = None
    if llm_enabled and (not force_no_llm):
        api_key = get_openai_key(key_file=api_key_file)
        if not api_key:
            logging.warning("LLM enabled but no API key found; skipping LLM refinement.")

    candidates = _load_candidates(candidates_jsonl)

    existing = {row["id"] for row in read_jsonl(out_jsonl)} if os.path.exists(out_jsonl) else set()
    doc_ids = [k for k in train_docs.keys() if k in candidates]
    if max_docs is not None:
        doc_ids = doc_ids[: max_docs]

    calls = 0

    # Open-model: dense encoder (bi-encoder) for fast silver labeling
    from data import parse_all

    cfg = load_config()
    d = parse_all(cfg.paths)
    class_ids, class_texts = _build_class_texts(d.id2name, d.keywords)
    cid2idx = {cid: i for i, cid in enumerate(class_ids)}

    # Initialize hierarchy for path-based label selection
    _init_hierarchy(d.child2parents, set(d.id2name.keys()))
    logging.info("Hierarchy initialized: %d leaves, %d parents", len(_LEAVES), len(_PARENT2CHILDREN))

    logging.info("Silver (embed) using %s", embed_model)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    encoder = SentenceTransformer(embed_model, trust_remote_code=embed_trust_remote_code, device=device)
    class_emb = encoder.encode(
        class_texts,
        batch_size=embed_batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    class_emb = np.asarray(class_emb, dtype=np.float32)

    reranker = None
    if use_reranker:
        logging.info("Reranker enabled: %s", reranker_model)
        reranker = CrossEncoder(reranker_model, trust_remote_code=reranker_trust_remote_code, device=device)

    rows_out: List[dict] = []

    # batch encode docs for speed
    pending: List[Tuple[str, str]] = []

    def _flush(batch: List[Tuple[str, str]]) -> None:
        nonlocal rows_out
        if not batch:
            return
        b_ids = [x for x, _ in batch]
        b_texts = [t for _, t in batch]
        doc_emb = encoder.encode(
            b_texts,
            batch_size=embed_batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        doc_emb = np.asarray(doc_emb, dtype=np.float32)

        # First pass: dense score on candidates for each doc
        per_doc_sorted: List[Tuple[str, List[str], List[float]]] = []
        for i, doc_id in enumerate(b_ids):
            cand_ids = [c for c in candidates.get(doc_id, []) if c in cid2idx]
            if not cand_ids:
                per_doc_sorted.append((doc_id, [], []))
                continue
            idxs = [cid2idx[c] for c in cand_ids]
            sims = (doc_emb[i] @ class_emb[idxs].T).astype(np.float32)
            order = np.argsort(-sims)
            sorted_ids = [cand_ids[j] for j in order]
            sorted_scores = [float(sims[j]) for j in order]
            per_doc_sorted.append((doc_id, sorted_ids, sorted_scores))

        # Optional second pass: rerank in one big batch for speed
        if reranker is not None:
            all_pairs: List[Tuple[str, str]] = []
            offsets: List[Tuple[int, int]] = []

            for doc_id, sorted_ids, sorted_scores in per_doc_sorted:
                if not sorted_ids:
                    offsets.append((len(all_pairs), len(all_pairs)))
                    continue

                if rerank_policy == "always":
                    need_rerank = True
                else:
                    # uncertain
                    if len(sorted_scores) >= 3:
                        margin23 = float(sorted_scores[1]) - float(sorted_scores[2])
                    else:
                        margin23 = 1.0
                    need_rerank = float(sorted_scores[1]) < low_conf or margin23 < small_margin

                if not need_rerank:
                    offsets.append((len(all_pairs), len(all_pairs)))
                    continue

                topn = min(rerank_top_n, len(sorted_ids))
                start = len(all_pairs)
                for c in sorted_ids[:topn]:
                    all_pairs.append((train_docs[doc_id], class_texts[cid2idx[c]]))
                end = len(all_pairs)
                offsets.append((start, end))

            if all_pairs:
                r_scores_all = reranker.predict(all_pairs)
                r_scores_all = np.asarray(r_scores_all, dtype=np.float32)
            else:
                r_scores_all = np.asarray([], dtype=np.float32)

            # Build outputs (and optionally queue for LLM refinement)
            llm_queue: List[LLMDoc] = []
            llm_base: Dict[str, List[str]] = {}

            for (doc_id, sorted_ids, sorted_scores), (start, end) in zip(per_doc_sorted, offsets):
                if not sorted_ids:
                    labels = class_ids[:2]
                    rows_out.append({"id": doc_id, "text": train_docs[doc_id], "labels": labels, "conf": 0.0})
                    continue

                if start == end:
                    final_ids = sorted_ids
                    final_scores = sorted_scores
                else:
                    r_scores = r_scores_all[start:end]
                    order = np.argsort(-r_scores)
                    final_ids = [sorted_ids[j] for j in order]
                    final_scores = [float(r_scores[j]) for j in order]

                # Use top-K selection (pure multi-label, no hierarchy enforcement)
                labels, conf = _pick_2_or_3(final_ids, final_scores, margin_for_third)
                llm_base[doc_id] = labels

                if api_key and llm_enabled and len(llm_queue) < llm_max_docs:
                    if llm_policy == "always":
                        need_llm = True
                    else:
                        if len(final_scores) >= 3:
                            margin23 = float(final_scores[1]) - float(final_scores[2])
                        else:
                            margin23 = 1.0
                        need_llm = float(final_scores[1]) < low_conf or margin23 < small_margin

                    if need_llm:
                        topn = min(int(llm_top_n), len(final_ids))
                        cand_pairs = [(cid, d.id2name.get(cid, "")) for cid in final_ids[:topn]]
                        llm_queue.append(LLMDoc(doc_id=doc_id, text=train_docs[doc_id], candidates=cand_pairs))

                rows_out.append({"id": doc_id, "text": train_docs[doc_id], "labels": labels, "conf": float(conf)})

            # Apply LLM refinement in batches, overwriting labels for queued docs
            if api_key and llm_enabled and llm_queue:
                calls_dir = os.path.join(llm_log_dir, "requests")
                for j in range(0, len(llm_queue), int(llm_docs_per_call)):
                    if calls >= max_calls:
                        break
                    chunk = llm_queue[j : j + int(llm_docs_per_call)]
                    refined = call_openai_batched(
                        api_key=api_key,
                        model=model,
                        temperature=llm_temperature,
                        docs=chunk,
                        calls_jsonl=os.path.join(llm_log_dir, "llm_calls.jsonl"),
                        calls_dir=calls_dir,
                        max_calls_total=max_calls,
                        rpm=rpm,
                    )
                    calls += 1
                    if not refined:
                        continue
                    # overwrite rows_out in-place for affected docs
                    refined_set = set(refined.keys())
                    for r in rows_out:
                        did = r.get("id")
                        if did in refined_set:
                            r["labels"] = refined[did]
                            r["conf"] = 1.0
        else:
            for doc_id, sorted_ids, sorted_scores in per_doc_sorted:
                if not sorted_ids:
                    labels = class_ids[:2]
                    rows_out.append({"id": doc_id, "text": train_docs[doc_id], "labels": labels, "conf": 0.0})
                    continue
                # Use top-K selection (pure multi-label, no hierarchy enforcement)
                labels, conf = _pick_2_or_3(sorted_ids, sorted_scores, margin_for_third)
                rows_out.append({"id": doc_id, "text": train_docs[doc_id], "labels": labels, "conf": float(conf)})

    for doc_id in tqdm(doc_ids, desc="silver_labeling", unit="doc"):
        if doc_id in existing:
            continue
        pending.append((doc_id, train_docs[doc_id]))
        if len(pending) >= 512:
            _flush(pending)
            pending = []

        if len(rows_out) >= 2000:
            prev = read_jsonl(out_jsonl) if os.path.exists(out_jsonl) else []
            write_jsonl(out_jsonl, prev + rows_out)
            rows_out = []

    _flush(pending)

    if rows_out:
        prev = read_jsonl(out_jsonl) if os.path.exists(out_jsonl) else []
        write_jsonl(out_jsonl, prev + rows_out)

    logging.info("Silver labeling done. New calls=%d, output=%s", calls, out_jsonl)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    cfg = load_config()

    parser = argparse.ArgumentParser()
    parser.add_argument("--max-docs", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--no-llm", action="store_true")
    parser.add_argument("--embed-model", type=str, default=cfg.retrieval.embed_model)
    parser.add_argument("--embed-trust-remote-code", action="store_true", default=cfg.retrieval.embed_trust_remote_code)
    parser.add_argument("--no-reranker", action="store_true")
    parser.add_argument("--reranker-model", type=str, default=cfg.retrieval.reranker_model)
    parser.add_argument("--reranker-trust-remote-code", action="store_true", default=cfg.retrieval.reranker_trust_remote_code)
    parser.add_argument("--rerank-top-n", type=int, default=cfg.retrieval.rerank_top_n)
    parser.add_argument("--rerank-policy", type=str, default=cfg.retrieval.rerank_policy, choices=["always", "uncertain"])
    parser.add_argument("--use-llm", action="store_true", default=cfg.llm.enabled)
    args = parser.parse_args()

    from data import parse_all

    d = parse_all(cfg.paths)

    if args.force and os.path.exists(cfg.paths.silver_file):
        os.remove(cfg.paths.silver_file)

    generate_silver(
        train_docs=d.train_docs,
        id2name=d.id2name,
        candidates_jsonl=cfg.paths.candidates_train,
        out_jsonl=cfg.paths.silver_file,
        llm_log_dir=cfg.paths.llm_dir,
        api_key_file=cfg.paths.openai_key_file,
        model=cfg.llm.model,
        max_calls=cfg.llm.max_calls,
        rpm=cfg.llm.max_rpm,
        max_docs=args.max_docs,
        force_no_llm=args.no_llm,
        embed_model=args.embed_model,
        embed_batch_size=cfg.retrieval.embed_batch_size,
        embed_trust_remote_code=bool(args.embed_trust_remote_code),
        use_reranker=(not args.no_reranker) and bool(cfg.retrieval.use_reranker),
        reranker_model=str(args.reranker_model),
        reranker_trust_remote_code=bool(args.reranker_trust_remote_code),
        rerank_top_n=int(args.rerank_top_n),
        rerank_policy=str(args.rerank_policy),
        low_conf=float(cfg.retrieval.low_conf),
        small_margin=float(cfg.retrieval.small_margin),
        margin_for_third=float(cfg.retrieval.margin_for_third),

        llm_enabled=bool(args.use_llm) and bool(cfg.llm.use_in_silver),
        llm_temperature=float(cfg.llm.temperature),
        llm_docs_per_call=int(cfg.llm.docs_per_call),
        llm_max_docs=int(cfg.llm.silver_max_docs),
        llm_policy=str(cfg.llm.silver_policy),
        llm_top_n=int(cfg.llm.silver_top_n),
    )


if __name__ == "__main__":
    main()
