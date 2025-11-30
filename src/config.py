import os

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

SEED = 42

for d in [EMB_DIR, MODEL_DIR, OUTPUT_DIR]:
    os.makedirs(d, exist_ok=True)
