# Structure-Aware Refinement for Hierarchical Multi-Label Text Classification

유한음
Korea University, Seoul, South Korea  
Student ID: 2021320045

## Abstract
Hierarchical multi-label text classification (HMTC) presents unique challenges, particularly when ground-truth labels are unavailable. In e-commerce product categorization, models must assign products to multiple categories within a taxonomy while adhering to parent-child relationships. In this work, we propose a three-stage pipeline for HMTC in an unlabeled setting.

Our approach consists of: (1) Hybrid Retrieval and Reranking to generate silver labels by combining dense and sparse retrieval with a candidate-union + dense reranking strategy, followed by a cross-encoder reranker; (2) GNN-Enhanced Logit Refinement, where a label-graph neural network explicitly models the taxonomy structure to refine classifier logits; and (3) Taxonomy-Consistent Inference, a decoding strategy that outputs an ancestry chain of length 2 or 3. Experimental results on the Amazon product dataset show that the structure-aware approach outperforms strong baselines, achieving a Samples-F1 score of **0.536**. In contrast, iterative Self-Training achieved **0.521**, suggesting that structural guidance is more reliable than repeated pseudo-labeling in this unlabeled setting.

## CCS Concepts
- Information systems $\rightarrow$ Retrieval models and ranking; Recommender systems.
- Computing methodologies $\rightarrow$ Natural language processing; Semi-supervised learning settings.

## Keywords
Hierarchical Classification, Silver Labeling, Graph Neural Networks, Logit Refinement, Hybrid Retrieval

## 1. Introduction
### 1.1 Background
Automated product categorization is critical for modern e-commerce platforms managing millions of items. The task is complicated by the hierarchical nature of product taxonomies, where a single item (e.g., “Baby Stroller”) must be classified into multiple related categories. Manual labeling at this scale is prohibitively expensive, motivating semi-supervised approaches that can leverage unlabeled data.

### 1.2 Problem Definition
Given an unlabeled corpus of product descriptions $D$ and a taxonomy graph $\mathcal{T} = (C, E)$, the goal is to learn a mapping $f: D \rightarrow \{0, 1\}^{|C|}$. Predictions must satisfy the hierarchy constraint: if a category $c$ is predicted ($y_c=1$), its ancestors must also be predicted ($y_{Pa(c)}=1$).

In this project setting, each document is constrained to output **exactly 2 or 3 labels**.

### 1.3 Contributions
1. **Hybrid Silver Labeling**: We generate silver labels by combining dense (SentenceTransformer) and sparse (TF-IDF) retrieval, taking the union of candidates and reranking that union using dense similarity, then applying a cross-encoder reranker for higher precision.
2. **GNN-Enhanced Architecture**: We introduce a label-graph neural module that refines the classifier’s output logits using taxonomy edges, correcting inconsistencies (e.g., high child score but low parent score).
3. **Taxonomy-Consistent Inference**: We output an ancestry chain of length 2 or 3 derived from the top-1 predicted label, with an optional extra leaf label when confidence is high, guaranteeing consistency with the taxonomy depth used in this dataset.

## 2. Related Work
### 2.1 Pseudo-Labeling Strategies
Semi-supervised learning often relies on self-training with high-confidence predictions [10]. Without initial labels, “cold start” is a major hurdle. We address this by casting label generation as an Information Retrieval (IR) problem, using the semantic content of category names and keywords as supervision.

### 2.2 Hierarchical Text Classification (HTC)
HTC approaches include local classifiers (one per node) and global approaches. Recent works use graph-based methods to incorporate hierarchy information. Our approach differs by applying message passing at the **logit refinement** stage, allowing the text encoder to focus on semantics while the graph module acts as a structural regularizer.

## 3. Methodology: High-Quality Silver Labeling
The quality of the final model is bounded by the quality of the training signal. We generate pseudo-labels via a structured silver-label pipeline.

### 3.1 Hybrid Retrieval Pipeline
We use a hybrid strategy:
- **Dense retrieval**: encode documents and categories using a dense encoder (e.g., BGE-M3) and rank via cosine similarity.
- **Sparse retrieval**: use TF-IDF over class-name-and-keyword text.
- **Hybrid fusion (implemented)**: take the union of TF-IDF and dense candidates, then rerank the union by dense similarity to select the final top-$k$ candidates.

### 3.2 Cross-Encoder Reranking and Label Selection
From the retrieved candidates, we optionally apply a cross-encoder reranker (bge-reranker-v2-m3) for higher precision. We then choose **2 or 3** labels using a margin rule:
- Select top-2 labels by default.
- Add a third label only if its reranker score is within a margin $\delta$ of the second label.

To reduce structural noise, we prefer selecting a leaf category and expanding it along the taxonomy to produce a short valid path of length up to 3.

### 3.3 Uncertainty-Aware LLM Refinement
To stay within a limited API budget (e.g., 1,000 calls), we apply uncertainty sampling and query the LLM only for ambiguous cases (low confidence or small margins). The LLM is constrained to **selection-only**: it must choose 2 or 3 labels from the provided candidate IDs.

Implementation note: prompts/call metadata are logged under `artifacts/llm_calls/` during runs, but these artifacts should not be committed to version control.

## 4. Methodology: GNN-Enhanced Classifier
[Figure 1 (place in Section 4 header): Overall architecture]
- Retrieval → silver labels → DeBERTa encoder → logits → label-GNN refinement → inference decoding
- Keep it as a single block diagram with arrows.

### 4.1 Backbone Encoder
We use DeBERTa-v3-base [2] as the text encoder. Given a document $d$, the [CLS] embedding $h_{CLS} \in \mathbb{R}^{768}$ summarizes the input.

### 4.2 Label-Graph Neural Refinement (Logit-Level)
We construct a graph $\mathcal{G}=(V, E)$ where nodes correspond to labels and edges come from parent-child relations (treated as bidirectional during message passing).

Initial logits are computed by a standard classifier head:

$$\hat{y}_{init} = W_{cls} h_{CLS} + b_{cls}$$

We then refine logits using message passing over the label graph with mean aggregation:

$$m_v = \mathrm{Mean}\{\phi(h_u) : (u \rightarrow v) \in E\}, \quad h_v' = \psi([h_v, m_v])$$

Finally, we blend the refined logits with the original logits via a learnable skip weight:

$$\hat{y}_{final} = \alpha \cdot \hat{y}_{refined} + (1 - \alpha) \cdot \hat{y}_{init}, \quad \alpha = \sigma(w)$$

This prevents over-smoothing while allowing structural correction.

### 4.3 Training Objectives
We address class imbalance with Focal Loss [6] and positive weighting (pos\_weight):

$$\mathcal{L} = -\alpha_t (1 - p_t)^\gamma \log(p_t)$$

We use $\gamma=2.0$ and pos\_weight=10.0.

## 5. Inference Strategy: Taxonomy-Consistent Decoding
Standard thresholding can yield structurally inconsistent predictions. Our inference outputs **exactly 2 or 3 labels** per document:
1. **Primary**: select the top-1 label by predicted probability.
2. **Ancestry expansion**: expand the selected label to its taxonomy path.
	- If the path length is 2, we keep the 2-label chain.
	- If the path length is 3, we keep the 3-label chain.
	- We do not extend to 4 labels.
3. **Optional extra leaf (only when 3 labels are required but the path has length 2)**: add the highest-probability leaf label not already included if its probability exceeds a threshold (0.4).
	- If no leaf exceeds the threshold, we fall back to the best remaining leaf to satisfy the “exactly 2 or 3 labels” requirement.

This yields hierarchy-consistent outputs for the taxonomy depth used in this dataset.

## 6. Experimental Results
### 6.1 Experimental Setup
- Dataset: Amazon products (1,945 train / 489 test)
- Metrics: Samples-F1 (Kaggle Evaluation)
- Implementation: PyTorch + HuggingFace Transformers, fixed random seed (42)

### 6.2 Main Results
Table 1 (place at start of Section 6.2): Performance comparison

| Method | Samples-F1 | Hierarchical Consistency |
|---|---:|:---:|
| Random (~uniform) | 0.01 | ✗ |
| DeBERTa-v3 + MLP (strong baseline) | 0.512 | ✗ |
| DeBERTa-v3 + Self-Training (iterative pseudo-labeling) | 0.521 | ✗ |
| DeBERTa-v3 + GNN Refinement (Ours) | **0.536** | ✓ |

Self-Training improved over the plain DeBERTa-v3 + MLP baseline but remained below the GNN-based approach. The gap (0.536 vs 0.521) can be explained by the interaction between pseudo-label noise and the learning dynamics of iterative relabeling. Our initial supervision comes from retrieval-based silver labels, which are high-recall but not perfectly clean. In Self-Training, the model is trained on this noisy signal and then re-labels the same data using its own predictions; this tends to create **Noise Accumulation** through **Confirmation Bias**, where early mistakes are reinforced because the model becomes increasingly confident in its own incorrect pseudo-labels. By contrast, the GNN module acts as **Structural Regularization**: it does not rely only on self-confidence, but uses the taxonomy graph as a **Taxonomy Prior** to suppress inconsistent outputs (e.g., predicting a child category with low parent support), which is especially beneficial when labels are imperfect.

## 7. Figure/Table Placement Guide
- Figure 1: Section 4 start (pipeline + model overview)
- Figure 2: Section 5 start (taxonomy-consistent decoding example on one sample)
- Table 1: Section 6.2 (main results)
- Table 2 (optional): Section 6.2 (ablation: no reranker / no GNN / different threshold)
- Figure 3 (optional): Section 7 (failure cases: silver noise vs semantic ambiguity)

## 7. Discussion
[Figure 2 (place at Section 7 start): Success vs failure cases]
- Show 1 success (hierarchy fixed) + 1 failure (wrong leaf due to silver noise).

The main qualitative difference between Self-Training and GNN-based refinement is how each method reacts to imperfect supervision.

**Why Self-Training underperformed (0.521).** Self-Training implicitly assumes that the model’s highest-confidence predictions are more accurate than the original pseudo-labels. In our setting, the initial silver labels are derived from retrieval and reranking, which can still contain semantic mismatches (especially for short or ambiguous descriptions). When the model re-labels the same data, it can amplify these mismatches (Confirmation Bias), leading to Noise Accumulation across iterations. As a result, the pseudo-label distribution drifts toward the model’s early errors.

**Why GNN refinement performed best (0.536).** The label-GNN introduces a taxonomy-based constraint during training and inference. When the text model produces an inconsistent set of logits, the GNN message passing redistributes confidence along parent-child relations, which stabilizes predictions under noisy silver supervision. This is a form of Structural Regularization driven by a Taxonomy Prior.

**Success case (structure fixed).** A document whose strongest logit is a specific leaf (e.g., “Baby Toys”) may have a weak parent logit due to sparse text cues. GNN refinement propagates confidence to the parent categories (e.g., “Baby Care”), making the final 2–3 label chain more consistent.

**Failure case (semantic error not fixable).** If the best leaf itself is wrong due to silver-label noise (e.g., “Electronics” vs “Automotive” ambiguity), the GNN can still produce a consistent ancestry chain, but it cannot correct the core semantic mistake without better supervision.

### 7.1 Undergraduate-Level Considerations (20 points)
1. **Why hybrid retrieval first?** Sparse TF-IDF captures keywords, while dense embeddings capture paraphrases. In an unlabeled setting, retrieval becomes the “teacher” signal for silver labels.
2. **Why union + dense rerank (not RRF)?** Union keeps recall; dense rerank restores semantic ordering without needing another fusion heuristic.
3. **Candidate set size tradeoff ($k$).** Larger $k$ increases recall but also increases reranker/LLM cost and noise. This pipeline uses a score-first approach: start wide, then rerank and select.
4. **Why add a cross-encoder reranker?** Bi-encoders are fast but less precise for close candidates. Cross-encoder reranking reduces systematic label swaps when categories have overlapping keywords.
5. **When does reranking help most?** Short texts and near-duplicate categories create small score margins; reranking helps resolve these “top-5 confusion” cases.
6. **Margin rule for 2 vs 3 labels.** Always outputting 3 increases noise; always outputting 2 misses legitimate multi-intent products. A margin-based third label is a simple uncertainty heuristic.
7. **Why prefer a leaf then expand?** The dataset’s labels follow a taxonomy; selecting a leaf gives specificity, and expansion guarantees hierarchical consistency.
8. **Taxonomy expansion vs pure thresholding.** Thresholding can output sibling leaves without parents; expansion enforces structure by construction.
9. **Why does inference allow an extra leaf at $p \ge 0.4$?** It spends the “third slot” only when a second leaf is confident enough, avoiding random third labels.
10. **Why GNN at the logit level?** It is model-agnostic: the text encoder predicts logits, and the label graph corrects inconsistencies without changing encoder internals.
11. **What does the label graph represent?** Nodes are classes; edges come from parent-child relations in `class_hierarchy.txt`. Message passing shares evidence between related labels.
12. **Why bidirectional edges?** Parent can support child and child can support parent; using both directions helps propagate confidence to ancestors and prevent orphan-child predictions.
13. **Skip connection motivation.** Pure GNN refinement can over-smooth, making many labels similar. A learnable skip weight keeps the raw classifier signal when the graph is not helpful.
14. **Learnable $\alpha$ vs fixed mix.** Fixed mixing assumes one best balance for all data. Learnable mixing adapts during training based on how noisy the graph signal is.
15. **Class imbalance handling.** Many categories are rare in silver labels. Focal loss and `pos_weight` reduce domination by frequent labels.
16. **Silver noise as the main bottleneck.** Errors from retrieval/reranking become training targets; the final model can only be as good as the silver labels.
17. **LLM refinement is selection-only.** This prevents label leakage and keeps outputs valid. It also makes evaluation safer because the LLM cannot invent unseen categories.
18. **LLM budget strategy.** Uncertainty sampling (low confidence, small margin) maximizes gain per call. Logging prompts/responses is essential for debugging and audit.
19. **Typical failure mode: semantically similar siblings.** Example: “home audio” vs “electronics accessories” style siblings. Even with hierarchy, picking the wrong leaf yields a consistent but incorrect path.
20. **Reproducibility vs speed tradeoff.** Fixed seeds improve comparability, but GPU nondeterminism and mixed precision can still cause small variations. Practical reporting should mention this.

### 7.2 Suggested 8-Page Layout (outline)
- Page 1: Abstract + Introduction
- Page 2: Related Work + Problem Setup
- Page 3: Silver Labeling (retrieval + reranking)
- Page 4: LLM refinement policy + cost controls
- Page 5: GNN logit refinement (equations + architecture)
- Page 6: Inference decoding + taxonomy consistency
- Page 7: Experiments + main results (Table 1)
- Page 8: Discussion (20 points) + Conclusion

## 8. Conclusion
We presented a pipeline for hierarchical multi-label classification without ground-truth labels. By combining hybrid retrieval for silver-label generation with GNN-based logit refinement and taxonomy-consistent decoding, we obtain robust performance under strict label-count constraints, reaching **0.536 Samples-F1**. A direct comparison with iterative Self-Training (0.521) suggests that structural guidance from the taxonomy is more effective than repeated self-relabeling when the initial pseudo-labels contain noise. Overall, **silver-label quality is the dominant bottleneck** in this unlabeled setting: improvements in retrieval/reranking that reduce semantic mismatches translate most directly into end-model gains.

## References
[1] Devlin, J., et al. (2019). BERT: Pre-training of Deep Bidirectional Transformers. NAACL.
[2] He, P., et al. (2021). DeBERTa: Decoding-enhanced BERT with Disentangled Attention. ICLR.
[3] Chen, J., et al. (2024). BGE M3-Embedding. arXiv.
[4] Zhou, J., et al. (2020). Hierarchy-aware global model for hierarchical text classification. ACL.
[5] Kipf, T. N., & Welling, M. (2017). Semi-Supervised Classification with Graph Convolutional Networks. ICLR.
[6] Lin, T. Y., et al. (2017). Focal Loss for Dense Object Detection. ICCV.
[7] Reimers, N., & Gurevych, I. (2019). Sentence-BERT. EMNLP.
[8] Nogueira, R., & Cho, K. (2019). Passage Re-ranking with BERT. arXiv.
[9] Sohn, K., et al. (2020). FixMatch: Simplifying Semi-Supervised Learning with Consistency and Confidence. NeurIPS.
[10] Lee, D. H. (2013). Pseudo-label: The simple and efficient semi-supervised learning method for deep neural networks. ICML Workshop.
