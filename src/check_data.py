# src/check_data.py
import json
import os

# 확인된 경로 반영
DATA_DIR = "./data/dataset"
CORPUS_FILE = os.path.join(DATA_DIR, "corpus.jsonl")
TAXONOMY_FILE = os.path.join(DATA_DIR, "product_categories.json")

def print_first_lines(filepath, n=2):
    print(f"\n=== Checking {os.path.basename(filepath)} ===")
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= n: break
                # jsonl 파일 처리
                if filepath.endswith('.jsonl'):
                    data = json.loads(line)
                    print(json.dumps(data, indent=2, ensure_ascii=False))
                # 일반 json 파일 처리
                else:
                    # json은 파일 전체를 읽어야 함 (너무 크면 앞부분만 읽기 위해 read 사용)
                    pass 
            
            # json 파일은 구조만 살짝 보기 위해 따로 처리
            if filepath.endswith('.json'):
                f.seek(0)
                content = json.load(f)
                # 딕셔너리면 키만 출력, 리스트면 첫 요소 출력
                if isinstance(content, dict):
                    print(f"Keys: {list(content.keys())[:5]}")
                    first_key = list(content.keys())[0]
                    print(f"Sample ({first_key}): {content[first_key]}")
                elif isinstance(content, list):
                    print(f"Sample item: {content[0]}")

    except Exception as e:
        print(f"Error reading file: {e}")

if __name__ == "__main__":
    print_first_lines(CORPUS_FILE)
    print_first_lines(TAXONOMY_FILE)