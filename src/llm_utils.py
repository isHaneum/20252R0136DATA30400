from __future__ import annotations

# 역할 role: llm calls
# 순서 order: optional refine
# 왜 why: pick 2-3 ids

import json
import logging
import os
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from openai import OpenAI

from utils import append_jsonl, ensure_dir, read_jsonl, utc_now_iso, write_text


@dataclass
class LLMDoc:
    doc_id: str
    text: str
    candidates: List[Tuple[str, str]]  # (class_id, class_name)


def get_openai_key(*, key_file: str) -> Optional[str]:
    key = os.getenv("OPENAI_API_KEY", "").strip()
    if key:
        return key
    if os.path.exists(key_file):
        key2 = open(key_file, "r", encoding="utf-8").read().strip()
        return key2 or None
    return None


def count_logged_calls(calls_jsonl: str) -> int:
    if not os.path.exists(calls_jsonl):
        return 0
    return len(read_jsonl(calls_jsonl))


def _make_batch_prompt(docs: Sequence[LLMDoc], *, max_chars: int = 900) -> str:
    lines: List[str] = []
    lines.append("Task: For each item, select 2 or 3 category IDs from the provided candidates.")
    lines.append("Rules:")
    lines.append("- You must choose ONLY from the candidate IDs shown for that item.")
    lines.append("- Output must be valid JSON ONLY (no markdown, no extra text).")
    lines.append("- Output format: a JSON array of objects, each {\"id\": <string>, \"labels\": [<id>,<id>(,<id>)]}.")
    lines.append("- Each labels list must contain exactly 2 or 3 IDs.")
    lines.append("")

    for doc in docs:
        text = doc.text.strip().replace("\r", " ")
        if len(text) > max_chars:
            text = text[:max_chars]
        lines.append(f"ID: {doc.doc_id}")
        lines.append("CANDIDATES:")
        for cid, name in doc.candidates:
            lines.append(f"- {cid}: {name}")
        lines.append("TEXT:")
        lines.append(text)
        lines.append("---")

    return "\n".join(lines)


def parse_batch_response(content: str, allowed_by_id: Dict[str, set[str]]) -> Dict[str, List[str]]:
    # Accept either a JSON array, or a dict with key 'items'.
    try:
        obj = json.loads(content)
    except Exception:
        return {}

    if isinstance(obj, dict) and "items" in obj:
        obj = obj["items"]

    if not isinstance(obj, list):
        return {}

    out: Dict[str, List[str]] = {}
    for item in obj:
        if not isinstance(item, dict):
            continue
        doc_id = str(item.get("id", "")).strip()
        labels = item.get("labels")
        if not doc_id or doc_id not in allowed_by_id:
            continue
        if not isinstance(labels, list):
            continue
        allowed = allowed_by_id[doc_id]
        picked: List[str] = []
        for x in labels:
            if not isinstance(x, str):
                continue
            x = x.strip()
            if x in allowed and x not in picked:
                picked.append(x)
        if len(picked) in (2, 3):
            out[doc_id] = picked
    return out


def call_openai_batched(
    *,
    api_key: str,
    model: str,
    temperature: float,
    docs: Sequence[LLMDoc],
    calls_jsonl: str,
    calls_dir: str,
    max_calls_total: int,
    rpm: int,
    timeout_s: int = 90,
) -> Dict[str, List[str]]:
    existing_calls = count_logged_calls(calls_jsonl)
    if existing_calls >= max_calls_total:
        logging.warning("LLM call cap reached (%d/%d). Skipping.", existing_calls, max_calls_total)
        return {}

    prompt = _make_batch_prompt(docs)
    call_id = f"{int(time.time())}_{existing_calls + 1}"

    # RPM throttle (coarse)
    min_interval = 60.0 / max(1, rpm)
    # Use last call timestamp if present
    last_ts = None
    if os.path.exists(calls_jsonl):
        rows = read_jsonl(calls_jsonl)
        if rows:
            last_ts = rows[-1].get("ts")
    # We don't parse last_ts; just sleep minimally to be safe
    time.sleep(min_interval)

    client = OpenAI(api_key=api_key, timeout=timeout_s)
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a careful classifier."},
            {"role": "user", "content": prompt},
        ],
        temperature=temperature,
    )
    content = resp.choices[0].message.content or ""

    ensure_dir(calls_dir)
    call_path = os.path.join(calls_dir, call_id)
    ensure_dir(call_path)
    write_text(os.path.join(call_path, "prompt.txt"), prompt)
    write_text(os.path.join(call_path, "response.txt"), content)

    # Log metadata
    ensure_dir(os.path.dirname(calls_jsonl))
    meta = {
        "call_id": call_id,
        "ts": utc_now_iso(),
        "model": model,
        "temperature": temperature,
        "n_docs": len(docs),
        "prompt_path": os.path.join(call_path, "prompt.txt"),
        "response_path": os.path.join(call_path, "response.txt"),
    }
    append_jsonl(calls_jsonl, [meta])

    allowed_by_id = {d.doc_id: {cid for cid, _ in d.candidates} for d in docs}
    return parse_batch_response(content, allowed_by_id)
