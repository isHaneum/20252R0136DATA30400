# Copilot instructions (project_llm)

## Big picture (data flow)
- Entry point: `run_pipeline.py` runs stages from `src/` with `PYTHONPATH=src`.
- Pipeline: retrieval candidates → taxonomy graph build → silver labeling (bi-encoder + optional reranker + optional LLM refinement) → multi-label training → inference → submission verification.
- Dataset is expected in `Amazon_products/` (within repo) by default; paths are centralized in `src/config.py` (`Paths`).

## How to run (Windows-friendly)
- PowerShell (recommended on Windows):
  - `python -m venv .venv`
  - `./.venv/Scripts/Activate.ps1`
  - `python -m pip install -r requirements.txt`
  - `python run_pipeline.py --student-id 2021320045`
- Bash runner (uses `.venv`, good for WSL/Git-Bash): `bash run.sh --student-id 2021320045`

## Key modules and what they own
- `src/data.py`: parses `classes.txt`, `class_hierarchy.txt`, `class_related_keywords.txt`, train/test corpora.
- `src/retrieval.py`: builds `artifacts/candidates_{train,test}.jsonl`.
  - Modes: `tfidf` / `dense` / `hybrid` (union then dense rerank). See `RetrievalConfig` in `src/config.py`.
- `src/silver_labeling.py`: writes `artifacts/silver_simple.jsonl` rows like `{id,text,labels,conf}`.
  - Always outputs **2 or 3 labels** per doc (course rule).
  - Optional OpenAI refinement is selection-only and must choose from candidates.
- `src/training.py`: trains HF `AutoModelForSequenceClassification` (multi-label) and saves a `label2id.json` mapping to prevent label index drift.
- `src/inference.py`: produces per-doc label strings and optionally refines ambiguous docs via OpenAI batching.
- `src/verify.py`: enforces submission schema + label validity + row count.

## LLM integration (must follow project constraints)
- LLM is optional; enabled via `--use-llm` flags on `run_pipeline.py` / stages.
- API key lookup order in `src/llm_utils.py`: `OPENAI_API_KEY` env var → `artifacts/llm_calls/openai.key`.
- Calls are batched (`docs_per_call`) and capped (`max_calls`), and every call logs:
  - `artifacts/llm_calls/llm_calls.jsonl` (metadata)
  - `artifacts/llm_calls/requests/<call_id>/{prompt.txt,response.txt}`
- Prompts require **JSON-only** responses and selection strictly from candidate IDs.

## Conventions and invariants to keep intact
- JSONL formats:
  - Candidates: `{ "id": <doc_id>, "candidates": [<class_id>...] }`
  - Silver: `{ "id", "text", "labels": [..], "conf": <float> }`
- Label count rule: every final output must contain **exactly 2 or 3** labels per doc.
- Taxonomy handling: inference expands a chosen leaf into a short parent→child path via `child2parents`.
- Important: `src/verify.py` currently validates header `pid,labels` with comma-separated labels, while `src/inference.py` writes header `id,label`. If you change either, keep them in sync or the pipeline will fail at the verify stage.

## Tests/status
- `tests/` exists but currently references modules not present in `src/` (e.g., `llm_labeling`, `postprocess`). Prefer validating changes by running `python run_pipeline.py` and `python src/verify.py` until tests are aligned.
