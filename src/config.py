from __future__ import annotations

import os
import torch
from dataclasses import dataclass, field


@dataclass
class Paths:
    project_root: str = field(default_factory=lambda: os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    data_dir: str = ""  # set in __post_init__

    artifacts_dir: str = ""  # set in __post_init__
    embeddings_dir: str = ""  # set in __post_init__
    model_dir: str = ""  # set in __post_init__
    output_dir: str = ""  # set in __post_init__

    classes_file: str = ""  # set in __post_init__
    hierarchy_file: str = ""  # set in __post_init__
    keyword_file: str = ""  # set in __post_init__
    train_corpus: str = ""  # set in __post_init__
    test_corpus: str = ""  # set in __post_init__

    candidates_train: str = ""  # set in __post_init__
    candidates_test: str = ""  # set in __post_init__

    silver_file: str = ""  # set in __post_init__

    llm_dir: str = ""  # set in __post_init__
    llm_calls_jsonl: str = ""  # set in __post_init__
    openai_key_file: str = ""  # set in __post_init__

    graph_file: str = ""  # set in __post_init__

    submission_file: str = ""  # set in __post_init__

    def __post_init__(self) -> None:
        self.project_root = os.path.abspath(self.project_root)
        self.data_dir = os.path.abspath(self.data_dir or os.path.join(self.project_root, "Amazon_products"))

        self.artifacts_dir = os.path.abspath(os.path.join(self.project_root, "artifacts"))
        self.embeddings_dir = os.path.abspath(os.path.join(self.artifacts_dir, "embeddings"))
        self.model_dir = os.path.abspath(os.path.join(self.project_root, "models"))
        self.output_dir = os.path.abspath(os.path.join(self.project_root, "output"))

        self.classes_file = os.path.join(self.data_dir, "classes.txt")
        self.hierarchy_file = os.path.join(self.data_dir, "class_hierarchy.txt")
        self.keyword_file = os.path.join(self.data_dir, "class_related_keywords.txt")
        self.train_corpus = os.path.join(self.data_dir, "train", "train_corpus.txt")
        self.test_corpus = os.path.join(self.data_dir, "test", "test_corpus.txt")

        self.candidates_train = os.path.join(self.artifacts_dir, "candidates_train.jsonl")
        self.candidates_test = os.path.join(self.artifacts_dir, "candidates_test.jsonl")

        self.silver_file = os.path.join(self.artifacts_dir, "silver_simple.jsonl")
        refined_silver = os.path.join(self.artifacts_dir, "silver_simple_refined.jsonl")
        if os.path.exists(refined_silver):
            self.silver_file = refined_silver

        self.llm_dir = os.path.join(self.artifacts_dir, "llm_calls")
        self.llm_calls_jsonl = os.path.join(self.llm_dir, "llm_calls.jsonl")
        self.openai_key_file = os.path.join(self.llm_dir, "openai.key")

        self.graph_file = os.path.join(self.artifacts_dir, "graph.json")

        student_id = os.getenv("STUDENT_ID", "2021320045")
        self.submission_file = os.path.join(self.output_dir, f"{student_id}_final.csv")

        # Additional paths from the other configuration
        self.train_emb_path = os.path.join(self.embeddings_dir, "train_emb.pt")
        self.test_emb_path = os.path.join(self.embeddings_dir, "test_emb.pt")
        self.label_emb_path = os.path.join(self.embeddings_dir, "label_emb.pt")
        self.best_model_path = os.path.join(self.model_dir, "best_model.pth")


@dataclass
class RetrievalConfig:
    # Score-first: larger candidate set improves recall.
    top_k: int = 200
    tfidf_max_features: int = 60000
    batch_docs: int = 512

    # tfidf | dense | hybrid (tfidf + dense union then dense rerank)
    mode: str = "hybrid"

    # Dense encoder for retrieval / silver (multilingual + strong)
    embed_model: str = "BAAI/bge-m3"
    embed_batch_size: int = 128
    embed_trust_remote_code: bool = False

    # For hybrid union size (per retriever before union)
    tfidf_top_k: int = 250
    dense_top_k: int = 250

    # Silver reranking (score-first)
    use_reranker: bool = True
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    reranker_trust_remote_code: bool = False
    rerank_top_n: int = 15  # Reduced from 60 for speed (top candidates usually correct)
    rerank_policy: str = "uncertain"  # Changed from "always" - only rerank uncertain cases

    # Uncertainty thresholds (used when rerank_policy=uncertain)
    low_conf: float = 0.35  # Increased threshold - more selective reranking
    small_margin: float = 0.05  # Increased threshold - more selective reranking

    # Selecting 2 vs 3 labels inside silver
    margin_for_third: float = 0.02


@dataclass
class LLMConfig:
    # LLM is optional; when enabled it must respect <=1000 total calls and log every prompt/response.
    enabled: bool = False
    model: str = "gpt-4o-mini"
    temperature: float = 0.0

    # hard cap for the course policy
    max_calls: int = 1000
    max_rpm: int = 450

    # docs per API call (batching is critical to stay within 1000 calls)
    docs_per_call: int = 10

    # Where to use the API
    use_in_silver: bool = True
    # Inference LLM refinement is disabled by default to avoid API calls during submission generation.
    use_in_inference: bool = False

    # For silver: only refine the hardest cases to spend calls wisely
    silver_max_docs: int = 6000
    silver_policy: str = "uncertain"  # uncertain | always
    silver_top_n: int = 40
    low_conf: float = 0.25
    small_margin: float = 0.03

    # For inference: refine only the most ambiguous docs
    infer_max_docs: int = 3000
    infer_policy: str = "uncertain"  # uncertain | always
    infer_top_n: int = 40


@dataclass
class TrainConfig:
    # Quality-first: DeBERTa v3 is typically stronger for this task.
    model_name: str = "microsoft/deberta-v3-base"
    max_length: int = 192
    # Keep VRAM/CPU safe by using a smaller batch with grad accumulation.
    batch_size: int = 16
    grad_accum_steps: int = 2  # effective batch ~= 32
    num_epochs: int = 8
    learning_rate: float = 3e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    seed: int = 42
    max_grad_norm: float = 1.0
    fp16: bool = True

    # Filter low-confidence silver labels
    # NOTE: Silver conf is mostly 0.0~0.3 (cosine sim). Use 0.0 to keep ALL data.
    min_confidence: float = 0.0  # Use all data; quality comes from reranker
    
    # DISABLED: Confidence weighting hurts more than helps when conf is low
    use_confidence_weight: bool = False


@dataclass
class InferConfig:
    batch_size: int = 16
    # Default selection is taxonomy-aware inference.py logic (when selection != dynamic).
    selection: str = "topk"  # topk | threshold | dynamic
    threshold: float = 0.6  # probability threshold (used by threshold/dynamic and taxonomy extra-leaf)
    drop_ratio: float = 0.65
    # Allow up to 4 labels by default (taxonomy path up to 3 + 1 extra leaf).
    min_labels: int = 2
    max_labels: int = 4
    leaf_threshold: float = 0.15  # lower threshold for leaf nodes (more specific)
    prefer_leaves: bool = True  # prioritize leaf nodes in output


@dataclass
class RunConfig:
    paths: Paths = field(default_factory=Paths)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    infer: InferConfig = field(default_factory=InferConfig)


def load_config() -> RunConfig:
    return RunConfig()
