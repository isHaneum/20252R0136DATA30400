import os
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import config
from utils import seed_everything, check_dir
from graph_build import build_adjacency_matrix
from models import GCNClassifier



#dataset class 정의
class GraphDataset(Dataset):
    def __init__(self, emb_path, label_dict, num_classes =513):


        #1. 문서 임베딩 로드
        data = torch.load(emb_path)
        self.pids = data['pids']
        self.embeddings = data['embeddings']
        #Pid to index
        self.pid2idx = {pid: idx for idx, pid in enumerate(self.pids)}

        #label 매핑

        self.indices = []#문서 index
        self.labels = []#

        for pid, label_ids in label_dict.items():# 
            if pid in self.pid2idx:
                self.indices.append(self.pid2idx[pid])

                #mulithot으로 0,1로 표현
                multi_hot = torch.zeros(num_classes)
                for lid in label_ids:
                    multi_hot[lid] = 1.0
                self.labels.append(multi_hot)
        

    def __len__(self):
        return len(self.indices) #문서 수

    def __getitem__(self, idx):
        #로드된 tensor에서
        doc_idx = self.indices[idx]
        return self.embeddings[doc_idx], self.labels[idx]



#학습 함수
def train_model(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0

    for docs, labels in loader:# 배치 단위
        docs = docs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        
        logits = model(docs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)#배치 평균 손실


def main():
    seed_everything(config.SEED)

    device = torch.device(config.DEVICE if config.DEVICE is not None else ('cuda' if torch.cuda.is_available() else 'cpu'))


    #1 adjacency matrix 생성
    num_classes = 531

    adj = build_adjacency_matrix().to(device)

    real_num_classes = adj.shape[0]
    print(f"Using {real_num_classes} classes.")
    #2 초기라벨 임베딩 load
    label_emb_path = os.path.join(config.EMB_DIR, "label_emb.pt")
    if os.path.exists(label_emb_path):
        label_init = torch.load(label_emb_path).to(device)
    
    else:
        label_init = torch.randn(real_num_classes, config.LABEL_EMB_DIM).to(device)

    #3. silver label load
    with open(config.SILVER_LABEL_PATH, 'r', encoding='utf-8') as f:
        silver_labels = json.load(f)

   
    #4. 모델 초기화
    model = GCNClassifier(doc_dim = 768, label_dim=768, adj=adj, num_classes=real_num_classes, label_init_emb=label_init).to(device)
    

    #loss, optimizer 설정, config에서 값 조절
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)

#phase 1: silver label로 사전학습
    print("1. Pretraining with Silver Labels")

    #train dataset, loader 생성
    train_dataset = GraphDataset(config.TRAIN_EMB_PATH, silver_labels, num_classes=real_num_classes)
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)

    #사전학습 루프
    for epoch in range(config.PRETRAIN_EPOCHS):
        avg_loss = train_model(model, train_loader, criterion, optimizer, device)
        print(f"Epoch {epoch+1}/{config.PRETRAIN_EPOCHS}, Loss: {avg_loss:.4f}")

#phase 2: 전체 훈련대이터 추론
    print("2. selftraing(pseudo labeling)")

    #dataset 다시 load x, 기존 tensor 사용
    train_emb_data = torch.load(config.TRAIN_EMB_PATH)

    #embeddings와 pids 가져오기
    all_embeddings = train_emb_data['embeddings'].to(device)
    all_pids = train_emb_data['pids']

    model.eval()
    pseudo_labels = silver_labels.copy()#기존 silver label 복사
    add_count = 0


    with torch.no_grad():
        batch_size = config.BATCH_SIZE
        for i in tqdm(range(0, all_embeddings.size(0), batch_size), desc="Generating Pseudo Labels"):
            #batch 단위로 처리
            batch_embs = all_embeddings[i:i+batch_size]
            batch_pids = all_pids[i:i+batch_size]
            #logits 계산, probs 계산(sigmoid)
            logits = model(batch_embs)
            probs = torch.sigmoid(logits)

            mask = probs > config.PSEUDO_LABEL_THRESHOLD

            #각 문서에 대해 확신하는 라벨 추가
            for j, is_confident in enumerate(mask):
                pid = all_pids[i+j]
            
                #기존에 label 없는 문서에 대해서만
                if pid not in pseudo_labels:
                    high_conf_idxs = torch.where(is_confident)[0].cpu().tolist()
                    if high_conf_idxs:
                        pseudo_labels[pid] = high_conf_idxs
                        add_count += 1

    print(f"Added {add_count} pseudo-labeled samples.")


    #5. 최종 전체 데이터로 모델 재학습
    print("3. Final Training with Silver + Pseudo Labels")

    if add_count > 0:
        #새로운 dataset, loader 생성
        final_dataset = GraphDataset(config.TRAIN_EMB_PATH, pseudo_labels, num_classes=real_num_classes)
        final_loader = DataLoader(final_dataset, batch_size=config.BATCH_SIZE, shuffle=True)

        #재학습 루프
        for epoch in range(config.PRETRAIN_EPOCHS):
            avg_loss = train_model(model, final_loader, criterion, optimizer, device)
            print(f"Epoch {epoch+1}/{config.PRETRAIN_EPOCHS}, Loss: {avg_loss:.4f}")
    
    check_dir(config.MODEL_DIR)
    model_save_path = os.path.join(config.MODEL_DIR, "final_gcn_model.pt")
    torch.save(model.state_dict(), model_save_path)
    print(f"✅ Final model saved to {model_save_path}")

if __name__ == "__main__":
    main()