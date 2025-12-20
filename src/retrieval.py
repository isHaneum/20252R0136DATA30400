from __future__ import annotations

import argparse
import json
import logging
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from sentence_transformers import SentenceTransformer

from config import load_config


@dataclass
class RetrievalResult:
    candidates_train: Dict[str, List[str]]
    candidates_test: Dict[str, List[str]]


def _build_class_texts(id2name: Dict[str, str], keywords: Dict[str, List[str]]) -> Tuple[List[str], List[str]]:
    class_ids = sorted(id2name.keys())
    texts: List[str] = []
    for cid in class_ids:
        name = id2name.get(cid, "")
        kw = keywords.get(cid, [])
        texts.append(" ".join([name] + kw))
    return class_ids, texts


def _topk(sim_row, class_ids: List[str], k: int) -> List[str]:
    if k >= len(class_ids):
        return class_ids[:]
    idx = sim_row.argsort()[::-1][:k]
    return [class_ids[i] for i in idx]


def build_candidates(
    *,
    id2name: Dict[str, str],
    keywords: Dict[str, List[str]],
    train_docs: Dict[str, str],
    test_docs: Dict[str, str],
    top_k: int,
    max_features: int = 50000,
) -> RetrievalResult:
    class_ids, class_texts = _build_class_texts(id2name, keywords)

    logging.info("TF-IDF fit on %d classes + %d docs", len(class_texts), len(train_docs) + len(test_docs))
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 2),
        stop_words=None,
        min_df=1,
    )

    doc_ids = list(train_docs.keys()) + list(test_docs.keys())
    doc_texts = [train_docs[k] for k in train_docs.keys()] + [test_docs[k] for k in test_docs.keys()]

    X = vectorizer.fit_transform(class_texts + doc_texts)
    X_class = X[: len(class_texts)]
    X_doc = X[len(class_texts) :]

    sims = cosine_similarity(X_doc, X_class)

    candidates_train: Dict[str, List[str]] = {}
    candidates_test: Dict[str, List[str]] = {}

    train_count = len(train_docs)
    for i, doc_id in enumerate(doc_ids):
        cands = _topk(sims[i], class_ids, top_k)
        if i < train_count:
            candidates_train[doc_id] = cands
        else:
            candidates_test[doc_id] = cands

    return RetrievalResult(candidates_train=candidates_train, candidates_test=candidates_test)


def build_candidates_dense(
    *,
    id2name: Dict[str, str],
    keywords: Dict[str, List[str]],
    train_docs: Dict[str, str],
    test_docs: Dict[str, str],
    top_k: int,
    embed_model: str,
    embed_batch_size: int,
    embed_trust_remote_code: bool = False,
) -> RetrievalResult:
    class_ids, class_texts = _build_class_texts(id2name, keywords)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info("Dense retrieval using %s on %s", embed_model, device)
    model = SentenceTransformer(embed_model, trust_remote_code=embed_trust_remote_code, device=device)

    class_emb = model.encode(
        class_texts,
        batch_size=embed_batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    class_emb = np.asarray(class_emb, dtype=np.float32)

    candidates_train: Dict[str, List[str]] = {}
    candidates_test: Dict[str, List[str]] = {}

    def _encode_and_rank(doc_items: List[Tuple[str, str]]) -> Dict[str, List[str]]:
        out: Dict[str, List[str]] = {}
        ids = [i for i, _ in doc_items]
        texts = [t for _, t in doc_items]
        doc_emb = model.encode(
            texts,
            batch_size=embed_batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,
        )
        doc_emb = np.asarray(doc_emb, dtype=np.float32)

        sims = doc_emb @ class_emb.T  # cosine since normalized
        for i, doc_id in enumerate(ids):
            out[doc_id] = _topk(sims[i], class_ids, top_k)
        return out

    candidates_train = _encode_and_rank(list(train_docs.items()))
    candidates_test = _encode_and_rank(list(test_docs.items()))
    return RetrievalResult(candidates_train=candidates_train, candidates_test=candidates_test)


def build_candidates_hybrid(
    *,
    id2name: Dict[str, str],
    keywords: Dict[str, List[str]],
    train_docs: Dict[str, str],
    test_docs: Dict[str, str],
    top_k: int,
    tfidf_top_k: int,
    dense_top_k: int,
    tfidf_max_features: int,
    embed_model: str,
    embed_batch_size: int,
    embed_trust_remote_code: bool = False,
) -> RetrievalResult:
    # Get TF-IDF candidates
    tfidf_res = build_candidates(
        id2name=id2name,
        keywords=keywords,
        train_docs=train_docs,
        test_docs=test_docs,
        top_k=tfidf_top_k,
        max_features=tfidf_max_features,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Precompute class embeddings once for hybrid
    class_ids, class_texts = _build_class_texts(id2name, keywords)
    model = SentenceTransformer(embed_model, trust_remote_code=embed_trust_remote_code, device=device)
    class_emb = model.encode(
        class_texts,
        batch_size=embed_batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    class_emb = np.asarray(class_emb, dtype=np.float32)
    cid2idx = {cid: i for i, cid in enumerate(class_ids)}

    # Get dense candidates (reuse same model + class embeddings)
    def _dense_from_precomputed(doc_dict: Dict[str, str]) -> Dict[str, List[str]]:
        out: Dict[str, List[str]] = {}
        doc_ids = list(doc_dict.keys())
        doc_texts = [doc_dict[i] for i in doc_ids]
        doc_emb = model.encode(
            doc_texts,
            batch_size=embed_batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,
        )
        doc_emb = np.asarray(doc_emb, dtype=np.float32)
        sims = doc_emb @ class_emb.T
        for i, doc_id in enumerate(doc_ids):
            out[doc_id] = _topk(sims[i], class_ids, dense_top_k)
        return out

    dense_train = _dense_from_precomputed(train_docs)
    dense_test = _dense_from_precomputed(test_docs)

    def _rerank(doc_dict: Dict[str, str], cands_a: Dict[str, List[str]], cands_b: Dict[str, List[str]]) -> Dict[str, List[str]]:
        out: Dict[str, List[str]] = {}
        doc_ids = list(doc_dict.keys())
        doc_texts = [doc_dict[i] for i in doc_ids]
        doc_emb = model.encode(
            doc_texts,
            batch_size=embed_batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,
        )
        doc_emb = np.asarray(doc_emb, dtype=np.float32)

        for i, doc_id in enumerate(doc_ids):
            union = list(dict.fromkeys((cands_a.get(doc_id, []) + cands_b.get(doc_id, []))))
            if not union:
                out[doc_id] = class_ids[:top_k]
                continue
            idxs = [cid2idx[c] for c in union if c in cid2idx]
            union_keep = [c for c in union if c in cid2idx]
            if not idxs:
                out[doc_id] = class_ids[:top_k]
                continue
            sims = (doc_emb[i] @ class_emb[idxs].T).astype(np.float32)
            order = np.argsort(-sims)[:top_k]
            out[doc_id] = [union_keep[j] for j in order]
        return out

    candidates_train = _rerank(train_docs, tfidf_res.candidates_train, dense_train)
    candidates_test = _rerank(test_docs, tfidf_res.candidates_test, dense_test)
    return RetrievalResult(candidates_train=candidates_train, candidates_test=candidates_test)


def write_candidates_jsonl(path: str, candidates: Dict[str, List[str]]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for doc_id, cands in candidates.items():
            f.write(json.dumps({"id": doc_id, "candidates": cands}, ensure_ascii=False) + "\n")
    logging.info("Wrote candidates: %s (%d docs)", path, len(candidates))


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    cfg = load_config()

    parser = argparse.ArgumentParser()
    parser.add_argument("--top-k", type=int, default=cfg.retrieval.top_k)
    parser.add_argument("--mode", type=str, default=cfg.retrieval.mode, choices=["tfidf", "dense", "hybrid"])
    parser.add_argument("--embed-model", type=str, default=cfg.retrieval.embed_model)
    parser.add_argument("--embed-trust-remote-code", action="store_true", default=cfg.retrieval.embed_trust_remote_code)
    args = parser.parse_args()

    from data import parse_all

    d = parse_all(cfg.paths)
    if args.mode == "tfidf":
        res = build_candidates(
            id2name=d.id2name,
            keywords=d.keywords,
            train_docs=d.train_docs,
            test_docs=d.test_docs,
            top_k=int(args.top_k),
            max_features=cfg.retrieval.tfidf_max_features,
        )
    elif args.mode == "dense":
        res = build_candidates_dense(
            id2name=d.id2name,
            keywords=d.keywords,
            train_docs=d.train_docs,
            test_docs=d.test_docs,
            top_k=int(args.top_k),
            embed_model=args.embed_model,
            embed_batch_size=cfg.retrieval.embed_batch_size,
            embed_trust_remote_code=bool(args.embed_trust_remote_code),
        )
    else:
        res = build_candidates_hybrid(
            id2name=d.id2name,
            keywords=d.keywords,
            train_docs=d.train_docs,
            test_docs=d.test_docs,
            top_k=int(args.top_k),
            tfidf_top_k=cfg.retrieval.tfidf_top_k,
            dense_top_k=cfg.retrieval.dense_top_k,
            tfidf_max_features=cfg.retrieval.tfidf_max_features,
            embed_model=args.embed_model,
            embed_batch_size=cfg.retrieval.embed_batch_size,
            embed_trust_remote_code=bool(args.embed_trust_remote_code),
        )
    write_candidates_jsonl(cfg.paths.candidates_train, res.candidates_train)
    write_candidates_jsonl(cfg.paths.candidates_test, res.candidates_test)


if __name__ == "__main__":
    main()
