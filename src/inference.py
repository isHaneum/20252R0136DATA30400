import os
import torch
import pandas as pd
import json
from tqdm import tqdm
from torch.utils.data import DataLoader

import config
from models import GCNClassifier
from graph_build import build_adjacency_matrix
from training import GraphDataset # 데이터셋 클래스 재사용

def main():
    print(">>> Step 6: Inference & Submission...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. 설정 로드
    adj = build_adjacency_matrix().to(device)
    real_num_classes = adj.shape[0]
    
    # 2. 모델 로드
    model = GCNClassifier(doc_dim=768, label_dim=768, adj=adj, num_classes=real_num_classes).to(device)

    model_path = config.BEST_MODEL_PATH  # config에서 경로 가져오기
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print("✅ Model weights loaded.")
    else:
        print("❌ Model weights not found. Please train first.")
        return

    # 3. 테스트 데이터 로드
    # 테스트용 데이터셋은 레이블이 없으므로 더미 레이블을 넣어 로드
    test_emb_path = os.path.join(config.EMB_DIR, "test_emb.pt")
    test_data = torch.load(test_emb_path)
    
    # 테스트 대상 ID 필터링 (존재하지 않을 경우 전체 테스트 임베딩의 PID 사용)
    test_pid_path = os.path.join(config.DATA_DIR, "category_classification", "pid2labelids_test.json")
    use_sequential_ids = False
    if os.path.exists(test_pid_path):
        with open(test_pid_path, 'r', encoding='utf-8') as f:
            target_pids = list(json.load(f).keys())
        output_ids = target_pids
        dummy_labels = {pid: [] for pid in target_pids}
    else:
        # 파일이 없으면 제출 형식(id=0..N-1)에 맞춰 순번 ID 사용
        use_sequential_ids = True
        output_ids = list(range(len(test_data['pids'])))
        # GraphDataset은 실제 PID로 인덱싱하므로 더미 라벨은 실제 PID로 작성
        dummy_labels = {pid: [] for pid in test_data['pids']}
    
    # 타겟 ID만 딕셔너리로 만듦 (GraphDataset 재활용을 위해)
    test_ds = GraphDataset(test_emb_path, dummy_labels, num_classes=real_num_classes)
    test_loader = DataLoader(test_ds, batch_size=config.BATCH_SIZE, shuffle=False)
    
    # 4. 예측
    model.eval()
    results = []
    
    print("Predicting...")
    with torch.no_grad():
        # GraphDataset은 (emb, label)을 반환하므로 label은 무시
        # 순서를 맞추기 위해 pid도 추적해야 함 -> test_ds.indices 순서대로 접근
        
        current_idx = 0
        for docs, _ in tqdm(test_loader):
            docs = docs.to(device)
            logits = model(docs)
            probs = torch.sigmoid(logits)
            
            # Threshold 0.5
            preds = (probs > 0.6).int().cpu().numpy()
            
            for pred_vec in preds:
                # 제출 형식에 맞춘 ID 선택
                pid = (output_ids[current_idx] if use_sequential_ids 
                       else test_ds.pids[test_ds.indices[current_idx]])
                
                # 1로 예측된 인덱스 추출
                indices = [str(i) for i, v in enumerate(pred_vec) if v == 1]
                
                # 하나도 예측 안 된 경우, Top 3
                if not indices:
                    top3 = torch.topk(probs[current_idx % preds.shape[0]], 3).indices.cpu().tolist()
                    indices = [str(i) for i in top3]
                
                # 라벨은 콤마로 구분: "3,21,56" 형태
                results.append({'id': pid, 'label': ",".join(indices)})
                current_idx += 1
                
    # 5. 저장
    submission_path = os.path.join(config.OUTPUT_DIR, "submission.csv")
    pd.DataFrame(results).to_csv(submission_path, index=False)
    print(f"🎉 Submission saved to {submission_path}")

if __name__ == "__main__":
    main()