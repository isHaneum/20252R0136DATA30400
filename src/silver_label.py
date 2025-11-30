import os
import json
from tqdm import tqdm
import config
from utils import seed_everything


def load_mapping_files():
    #text 파일들을 읽어서 dic으로 변환
    # 1. ClassId2Name
    label2id = {}

    with open(config.CLASSES_PATH, 'r', encoding='utf-8', errors='ignore') as f:
        
        for line in f:
            parts = line.strip().split()
            
            if len(parts) >= 2:
                lid = int(parts[0])
                name = " ".join(parts[1:])
                label2id[name.lower()] = lid

    # 2. 키워드 to label Id
    keyword2id = {}
    with open(config.KEYWORDS_PATH, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            parts = line.strip().split(':')
            # 데이터형식 dataset: class_name:kw1,kw2,...
            if len(parts) >= 2:
                cname = parts[0].strip()
                keywords_str = ':'.join(parts[1:])
                lid = label2id.get(cname.lower())
                if lid is None:
                    # try without lower (defensive)
                    lid = label2id.get(cname)
                if lid is None:
                    # class name not found in classes.txt -> skip
                    continue

                for kw in keywords_str.split(','):
                    clean_kw = kw.strip().lower()
                    if clean_kw:
                        keyword2id[clean_kw] = lid

    # 3. Taxonomy (child -> parent)
    child_to_parent = {}
    with open(config.TAXONOMY_PATH, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            parts = line.strip().split()# 공백 기준

            if len(parts) >= 2:
                parent, child = int(parts[0]), int(parts[1])
                if child not in child_to_parent:
                    child_to_parent[child] = set()# 중복 방지
                child_to_parent[child].add(parent)

    return label2id, keyword2id, child_to_parent





def main():# silver label 생성
    seed_everything(config.SEED)
    print("Generating Silver Labels")

    label2id, keyword2id, child_to_parent = load_mapping_files()
    print(f"Loaded {len(keyword2id)} keywords")#키워드 수 확인

    silver_labels = {}
    
    # 훈련 데이터
    with open(config.TRAIN_CORPUS_PATH, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()

    for line in tqdm(lines):
        parts = line.strip().split('\t', 1)
        if len(parts) < 2:
            continue

        pid = parts[0]  # 첫 번째는 Product ID
        text = parts[1].lower()  # 나머지는 텍스트 (keep original spacing)
        
        matched = set()
        
        # 1. 키워드 매칭
        for kw, lid in keyword2id.items():
            if kw in text:
                matched.add(lid)
        
        # 2. 상위 카테고리 자동 추가
        queue = list(matched) #queue는 bfs용
        visited = set(matched) #방문한 노드 추적
        
        while queue:
            curr = queue.pop(0)

            if curr in child_to_parent:
                for parent in child_to_parent[curr]:
                    if parent not in visited:
                        visited.add(parent)
                        queue.append(parent)
                        matched.add(parent)
        
        if matched:
            silver_labels[pid] = list(matched)

    # 결과 저장
    with open(config.SILVER_LABEL_PATH, 'w', encoding='utf-8') as f:
        json.dump(silver_labels, f)
    print(f"✅ Saved {len(silver_labels)} silver labels to {config.SILVER_LABEL_PATH}")# 확인

if __name__ == "__main__":
    main()