# project_llm Pipeline Audit (2025-12-20)

**최종 업데이트**: 2025-12-20
**Student ID**: 2021320045

## 핵심 변경사항 (v2)
- ✅ Inference threshold: 0.6 → **0.4** 변경
- ✅ Colab A100 GPU 최적화 노트북 생성: `colab_pipeline.ipynb`
- ✅ 불필요한 중간 파일 정리 (silver_full, silver_leaf, silver_seed 등)
- ✅ Gemini API 토큰/Rate Limit 정보 확인 및 최적화

---

This document maps **exactly** which code produces which artifacts, how silver labels are generated, how/when the LLM is called (with counts), and how the GNN is defined/used in training and inference.

## 1) End-to-end flow (run_pipeline.py)
Entry: [run_pipeline.py](run_pipeline.py)

Stages (executed from `src/` with `PYTHONPATH=src`):
1. Retrieval candidates → [src/retrieval.py](src/retrieval.py)
   - Output:
     - `artifacts/candidates_train.jsonl`
     - `artifacts/candidates_test.jsonl`
2. Taxonomy graph build → [src/graph_build.py](src/graph_build.py)
   - Output:
     - `artifacts/graph.json` (nodes+edges built from `child2parents`)
3. Silver labeling → [src/silver_labeling.py](src/silver_labeling.py)
   - Input: `artifacts/candidates_train.jsonl` + raw train corpus
   - Output:
     - `artifacts/silver_simple.jsonl` (or `artifacts/silver_simple_refined.jsonl` if created)
4. Training (baseline) → [src/training.py](src/training.py)
   - Input: `artifacts/silver_simple*.jsonl`
   - Output:
     - `models/student/` (HF model + tokenizer)
     - `models/student/label2id.json`
5. Inference (baseline) → [src/inference.py](src/inference.py)
   - Input: `models/student/` + `artifacts/candidates_test.jsonl` + taxonomy
   - Output:
     - `output/<STUDENT_ID>_final.csv`

Optional GNN branch:
- Training (GNN) → [src/gnn_classifier.py](src/gnn_classifier.py)
- Inference (GNN) → [src/gnn_inference.py](src/gnn_inference.py)

## 2) Submission output format
Required output CSV format:
- Header: `id,label`
- `label` is a **single cell** containing comma-separated label IDs, e.g. `0,"93,43,181"`.

Writer behavior:
- Both [src/inference.py](src/inference.py) and [src/gnn_inference.py](src/gnn_inference.py) use `csv.writer`, so the `label` cell is quoted when needed and Excel stays 2 columns.

Verifier:
- [src/verify.py](src/verify.py) enforces:
  - header exactly `id,label`
  - exactly 2 or 3 labels per row
  - label IDs must exist in `Amazon_products/classes.txt`

## 3) Inference label selection rule (baseline + GNN)
Implemented in:
- baseline: `_select_taxonomy_labels(...)` in [src/inference.py](src/inference.py)
- GNN: `_select_labels_from_probs(...)` in [src/gnn_inference.py](src/gnn_inference.py)

Rule (as implemented):
1. Compute top-1 predicted label `primary`.
2. Expand its taxonomy ancestry to length up to 3:
   - Prefer `[grandparent, parent, primary]` (top-down).
3. If ancestry yields fewer than 3 labels, add **exactly one** extra label whose probability is `>= 0.4` (threshold 변경됨, 기존 0.6).
4. Always return exactly 2 or 3 labels (pad by duplication if necessary).

## 4) Silver label generation (silver_simple.jsonl)
Code: [src/silver_labeling.py](src/silver_labeling.py)

Inputs:
- Train docs + taxonomy from [src/data.py](src/data.py) via `parse_all(...)`
- Candidate lists: `artifacts/candidates_train.jsonl`
- Class names: `Amazon_products/classes.txt`
- Related keywords: `Amazon_products/class_related_keywords.txt`

Core logic (high level):
1. For each training doc id, read its candidate IDs from `candidates_train.jsonl`.
2. Score candidates using embedding similarity (SentenceTransformer) and optionally rerank using a CrossEncoder.
3. Pick 2 or 3 labels using margin rule (`_pick_2_or_3`) or taxonomy-path pick (`_pick_hierarchy_path`).
4. Optionally refine uncertain cases with the LLM (OpenAI), **selection-only** (must choose from provided candidate IDs).

LLM prompt used (silver):
- Single-doc prompt template: `_make_prompt(...)` inside [src/silver_labeling.py](src/silver_labeling.py)
- Batched prompt template: `_make_batch_prompt(...)` inside [src/llm_utils.py](src/llm_utils.py)

## 5) LLM usage audit (from artifacts)
Log folder: `artifacts/llm_calls/`

Observed on this workspace:
- `artifacts/llm_calls/llm_calls.jsonl`: **226** lines
- `artifacts/llm_calls/leaf_silver_calls.jsonl`: **1000** lines
- `artifacts/llm_calls/calls.jsonl`: **1** line
- `artifacts/llm_calls/requests/`: **missing**

Important implication:
- The code in [src/llm_utils.py](src/llm_utils.py) *would* save per-call prompts/responses under `artifacts/llm_calls/requests/<call_id>/{prompt.txt,response.txt}`.
- In this workspace run, that `requests/` directory is not present, so prompts/responses are not recoverable from disk (only call metadata counts are available).

## 6) GNN model: what it is and how it’s used
Code: [src/gnn_classifier.py](src/gnn_classifier.py)

What the GNN is (definition):
- Text encoder: `AutoModel` (Transformer) produces a CLS embedding.
- Classifier head produces initial logits over all labels.
- **LabelGNNLayer** refines logits using a label graph (taxonomy edges).

Nodes/edges:
- Nodes correspond to **labels/classes** (not documents).
- Edges come from taxonomy relations (parent→child) passed as `edge_index`.

Skip connections:
- In `LabelGNNLayer.forward`:
  - internal skip: `h2 = fc2(h) + h`
  - output skip/blend: `output = alpha * refined + (1 - alpha) * original_logits` with learnable `skip_weight`.

Layer count:
- The label refinement module is a 2-layer MLP/message-passing style block (fc1 + message passing + fc2 + fc_out) with skip connections.

Inference:
- GNN inference uses the trained GNN model to produce logits → sigmoid probabilities → applies the exact same taxonomy selection rule described in section 3.

## 7) Gemini API Token/Rate Limit 정보

### 권장 모델: gemini-2.0-flash
- **Free Tier**: 무료 (Standard)
- **Context Window**: 1,000,000 tokens
- **Token 계산**: 1 token ≈ 4 characters, 100 tokens ≈ 60-80 English words

### Rate Limits (Free Tier)
- RPM (Requests per Minute): 15
- TPM (Tokens per Minute): 1,000,000
- RPD (Requests per Day): 1,500

### 최적 배치 전략
- `docs_per_call`: 30 (토큰 효율성)
- `max_calls`: 500 (안전 마진)
- 텍스트 truncation: 300자 (토큰 절약)

---

## 8) Colab A100 최적화 설정

파일: `colab_pipeline.ipynb`

### A100 GPU 최적화 파라미터
```python
{
    "batch_size": 64,           # A100 80GB 메모리 활용
    "embed_batch_size": 256,    # 대용량 임베딩 배치
    "max_length": 256,          # 긴 컨텍스트
    "num_epochs": 10,           # 충분한 학습
    "learning_rate": 2e-5,
    "fp16": True,               # Mixed precision
}
```

---

## 9) 최종 산출물 목록

### 필수 파일
- `artifacts/candidates_train.jsonl` - 학습 후보
- `artifacts/candidates_test.jsonl` - 테스트 후보
- `artifacts/graph.json` - Taxonomy 그래프
- `artifacts/silver_simple.jsonl` - Silver 라벨
- `student_gnn/` - 학습된 GNN 모델
- `output/2021320045_final.csv` - 최종 제출 파일

### 삭제된 파일 (불필요)
- ~~silver_full.jsonl~~
- ~~silver_leaf.jsonl~~
- ~~silver_seed.jsonl~~
- ~~silver_seed_fixed.jsonl~~
- ~~train_sweep.jsonl~~
- ~~sanity_report.txt~~
