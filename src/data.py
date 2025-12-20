from __future__ import annotations

# 역할 role: data parse
# 순서 order: shared input
# 왜 why: unify formats

import logging
from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass
class ParsedData:
    id2name: Dict[str, str]
    child2parents: Dict[str, List[str]]
    parent2children: Dict[str, List[str]]
    keywords: Dict[str, List[str]]
    train_docs: Dict[str, str]
    test_docs: Dict[str, str]


def load_classes(path: str) -> Dict[str, str]:
    id2name: Dict[str, str] = {}
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or "\t" not in line:
                continue
            cid, name = line.split("\t", 1)
            cid, name = cid.strip(), name.strip()
            if cid and name:
                id2name[cid] = name
    return id2name


def load_hierarchy(path: str) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
    child2parents: Dict[str, List[str]] = {}
    parent2children: Dict[str, List[str]] = {}
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or "\t" not in line:
                continue
            parent, child = line.split("\t", 1)
            parent, child = parent.strip(), child.strip()
            if not parent or not child:
                continue
            child2parents.setdefault(child, []).append(parent)
            parent2children.setdefault(parent, []).append(child)
    return child2parents, parent2children


def load_keywords(path: str, *, name2id: Dict[str, str]) -> Dict[str, List[str]]:
    keywords: Dict[str, List[str]] = {}
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or ":" not in line:
                continue
            key, rest = line.split(":", 1)
            key = key.strip()
            cid = name2id.get(key)
            kw = [x.strip() for x in rest.split(",") if x.strip()]
            if cid:
                keywords[cid] = kw
    return keywords


def load_train_corpus(path: str) -> Dict[str, str]:
    docs: Dict[str, str] = {}
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.rstrip("\n")
            if not line.strip() or "\t" not in line:
                continue
            doc_id, text = line.split("\t", 1)
            doc_id, text = doc_id.strip(), text.strip()
            if doc_id and text:
                docs[doc_id] = text
    return docs


def load_test_corpus(path: str) -> Dict[str, str]:
    docs: Dict[str, str] = {}
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.rstrip("\n")
            if not line.strip() or "\t" not in line:
                continue
            doc_id, text = line.split("\t", 1)
            doc_id, text = doc_id.strip(), text.strip()
            if doc_id and text:
                docs[doc_id] = text
    return docs


def parse_all(paths) -> ParsedData:
    logging.info("Loading dataset from %s", paths.data_dir)
    id2name = load_classes(paths.classes_file)
    name2id = {v: k for k, v in id2name.items()}
    child2parents, parent2children = load_hierarchy(paths.hierarchy_file)
    keywords = load_keywords(paths.keyword_file, name2id=name2id)
    train_docs = load_train_corpus(paths.train_corpus)
    test_docs = load_test_corpus(paths.test_corpus)
    return ParsedData(
        id2name=id2name,
        child2parents=child2parents,
        parent2children=parent2children,
        keywords=keywords,
        train_docs=train_docs,
        test_docs=test_docs,
    )
