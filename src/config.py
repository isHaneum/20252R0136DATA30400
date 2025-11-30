import os
import torch

# 기본 경로 설정
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "Amazon_products")
EMB_DIR = os.path.join(BASE_DIR, "data", "embeddings")
MODEL_DIR = os.path.join(BASE_DIR, "models")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

# 데이터 파일 경로
TRAIN_CORPUS_PATH = os.path.join(DATA_DIR, "train", "train_corpus.txt")
TEST_CORPUS_PATH = os.path.join(DATA_DIR, "test", "test_corpus.txt")
CLASSES_PATH = os.path.join(DATA_DIR, "classes.txt")
TAXONOMY_PATH = os.path.join(DATA_DIR, "class_hierarchy.txt")
KEYWORDS_PATH = os.path.join(DATA_DIR, "class_related_keywords.txt")

# 실버 라벨 저장 경로
SILVER_LABEL_PATH = os.path.join(DATA_DIR, "silver_labels.json")

# Embedding files (defaults)
TRAIN_EMB_PATH = os.path.join(EMB_DIR, "train_emb.pt")
TEST_EMB_PATH = os.path.join(EMB_DIR, "test_emb.pt")
LABEL_EMB_PATH = os.path.join(EMB_DIR, "label_emb.pt")

# reproducibility
SEED = 42

for d in [EMB_DIR, MODEL_DIR, OUTPUT_DIR]:
    os.makedirs(d, exist_ok=True)

# Hardware
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS = 4

# Model / tokenizer
MODEL_NAME = "bert-base-uncased"
LABEL_EMB_DIM = 768
MAX_LEN = 256

# Training hyperparameters
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
BATCH_SIZE = 128
PRETRAIN_EPOCHS = 5
TRAIN_EPOCHS = 5
WARMUP_STEPS = 0
GRAD_CLIP_NORM = 1.0
DROPOUT = 0.2

# Pseudo-label / self-training
PSEUDO_LABEL_THRESHOLD = 0.9

# Paths
BEST_MODEL_PATH = os.path.join(MODEL_DIR, "best_model.pth")