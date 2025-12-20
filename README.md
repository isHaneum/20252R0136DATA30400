# Structure-Aware Refinement for Hierarchical Multi-Label Text Classification

**GNN-Enhanced Amazon Product Categorization Pipeline**

This repository implements a complete pipeline for hierarchical multi-label text classification using GNN-enhanced logit refinement. The system classifies Amazon product descriptions into a 31-category taxonomy without ground-truth labels.

---

## Pipeline Blueprint

```
[Phase 1: Silver Label Generation]
Train Docs → Hybrid Retrieval (Dense + TF-IDF) → Top-k Candidates
    → Cross-Encoder Reranking (bge-reranker-v2-m3) → Margin Selection (2-3 labels)
    → Optional LLM Refinement (low confidence only) → Silver Labels

[Phase 2: Model Training]
Silver Labels + Taxonomy Graph → DeBERTa-v3 Encoder
    → Linear Classifier → Label-Graph Logit Refinement (GNN)
    → Learnable Skip Blend → Focal Loss Training → Trained Model

[Phase 3: Inference]
Test Docs → Trained Model → Probability Vector
    → Best Leaf Selection → Taxonomy Path Expansion [Leaf, Parent, Grandparent]
    → Optional Extra Leaf Addition (if prob ≥ 0.4) → Final Prediction (2-3 labels)
```

**Key Design Principles:**
- LLM calls are **optional** and **selection-only** (never generates new labels)
- LLM is used only for **low-confidence silver labeling**, not during inference
- All predictions respect **hierarchical consistency** via path expansion
- Label count is **exactly 2 or 3** per document

---

## Project Structure

```
project_llm/
├── src/                          # Source code
│   ├── retrieval.py              # Hybrid retrieval (Dense + TF-IDF)
│   ├── silver_labeling.py        # Silver label generation with reranking
│   ├── gnn_classifier.py         # GNN-enhanced multi-label classifier
│   ├── gnn_inference.py          # Taxonomy-aware inference
│   ├── verify.py                 # Submission verification
│   ├── config.py                 # Configuration management
│   ├── data.py                   # Data loading utilities
│   ├── llm_utils.py              # LLM API integration
│   └── utils.py                  # Helper functions
├── Amazon_products/              # Dataset
│   ├── classes.txt               # Category definitions (31 classes)
│   ├── class_hierarchy.txt       # Taxonomy structure
│   ├── class_related_keywords.txt
│   ├── train/train_corpus.txt    # Training documents
│   └── test/test_corpus.txt      # Test documents
├── artifacts/                    # Generated artifacts (created at runtime; not committed)
├── student_gnn/                  # Trained GNN model (created at runtime; not committed)
├── output/                       # Final predictions (created at runtime; not committed)
├── run.sh                        # Full pipeline script (bash)
├── run_pipeline.py               # Pipeline runner (Python)
├── requirements.txt              # Dependencies
└── FINAL_REPORT.md               # Technical report (8 pages)
```

---

## Quick Start

### 1. Environment Setup

```bash
# Create virtual environment
python -m venv .venv

# Activate (Linux/Mac)
source .venv/bin/activate

# Activate (Windows PowerShell)
.\.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Full Pipeline

**Option A: Using bash script**
```bash
bash run.sh --student-id 2021320045
```

**Option B: Using Python runner**
```bash
python run_pipeline.py --student-id 2021320045
```

### 3. Run Individual Stages

```bash
# Stage 1: Hybrid Retrieval (Dense + TF-IDF)
python src/retrieval.py

# Stage 2: Silver Labeling (Cross-Encoder Reranking)
python src/silver_labeling.py

# Stage 3: GNN Training
python src/gnn_classifier.py --epochs 10 --batch-size 32 --lr 2e-5

# Stage 4: Taxonomy-Aware Inference
python src/gnn_inference.py --model-dir student_gnn --output output/2021320045_final.csv

# Stage 5: Verification
python src/verify.py
```

---

## Pipeline Architecture

### Stage 1: Hybrid Retrieval
- **Dense Encoder**: BAAI/bge-m3 for semantic similarity
- **Sparse Encoder**: TF-IDF for keyword matching
- **Fusion (implemented)**: union of dense + TF-IDF candidates, then dense similarity reranking over the union
- **Output**: Top-50 candidate categories per document

### Stage 2: Silver Labeling
- **Cross-Encoder**: BAAI/bge-reranker-v2-m3 for precise ranking
- **Selection**: Margin-based 2-3 label selection
- **LLM Refinement**: Optional OpenAI API for ambiguous cases (selection-only)
- **Output**: High-quality pseudo-labels

### Stage 3: GNN-Enhanced Training
- **Text Encoder**: DeBERTa-v3-base (frozen + last 2 layers fine-tuned)
- **GNN Module**: 2-layer Label-GNN with skip connections
- **Loss**: Focal Loss + Positive Weighting (pos_weight=10.0)
- **Output**: Trained classifier with taxonomy awareness

### Stage 4: Taxonomy-Aware Inference
- **Path Expansion**: [Primary → Parent → Grandparent] (up to 3 labels)
- **Extra Label**: Add one leaf if probability ≥ 0.4 (only when outputting 3 labels)
- **Output**: Exactly 2 or 3 hierarchically consistent labels per document

---

## Configuration

Key hyperparameters (configurable via CLI or `src/config.py`):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `encoder` | `microsoft/deberta-v3-base` | Text encoder model |
| `batch_size` | 32 | Training batch size |
| `learning_rate` | 2e-5 | Learning rate |
| `epochs` | 10 | Number of training epochs |
| `max_length` | 192 | Max sequence length |
| `threshold` | 0.4 | Extra leaf selection threshold |
| `gnn_hidden` | 256 | GNN hidden dimension |

---

## Reproducibility

All random seeds are fixed:
```python
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
```

---

## LLM API Logs

All LLM API calls are logged under `artifacts/llm_calls/`:
- `llm_calls.jsonl`: Metadata (timestamps, model, token counts)
- `requests/<call_id>/prompt.txt`: Input prompts
- `requests/<call_id>/response.txt`: Model responses

Note: these logs can contain sensitive information (prompts/metadata). Do not commit them.

---

## External Resources

Large files are available via Google Drive (not committed to GitHub):

- **Artifacts** (candidates/silver/LLM logs produced by runs):
    - https://drive.google.com/drive/folders/1iGQ1zRRA-52wA_ANZnFt_P1-SPj4ik6P?usp=sharing
    - Place into: `project_llm/artifacts/`
- **Models** (optional, legacy/extra checkpoints):
    - https://drive.google.com/drive/folders/1JNVaGedbOfWcbMUG4EY-0I2FQx7Jz1mo?usp=drive_link
    - Place into: `project_llm/models/` (only needed if you use other scripts/checkpoints)
- **student_gnn** (required if you want to run inference without retraining):
    - https://drive.google.com/drive/folders/1LCzwxno4xmSDOTjsOolC_PZAsimf0beH?usp=drive_link
    - Place into: `project_llm/student_gnn/`

If you download `student_gnn/`, you can run:
```bash
python src/gnn_inference.py --model-dir student_gnn --student-id 2021320045
python src/verify.py
```

---

## Experimental Results

| Method | Samples-F1 | Description |
|--------|------------|-------------|
| Random (~uniform) | 0.01 | Random label assignment |
| DeBERTa-v3 + MLP (strong baseline) | 0.512 | Flat classifier |
| DeBERTa-v3 + Self-Training | 0.521 | Iterative pseudo-labeling |
| **DeBERTa-v3 + GNN Refinement (Ours)** | **0.536** | Taxonomy-aware refinement |

---

## Submission Format

Output CSV (`output/<STUDENT_ID>_final.csv`):
- **Header**: `id,label`
- **Labels**: Comma-separated category IDs (2 or 3 per document)

Example:
```csv
id,label
0,"5,12,28"
1,"3,7"
2,"1,4,15"
```

---

## References

1. He et al., "DeBERTa: Decoding-enhanced BERT with Disentangled Attention", ICLR 2021
2. Chen et al., "BGE M3-Embedding", 2024
3. Kipf & Welling, "Semi-Supervised Classification with GCNs", ICLR 2017
4. Lin et al., "Focal Loss for Dense Object Detection", ICCV 2017

---

**Student ID**: 2021320045  
**Course**: LLM Applications (December 2024)
