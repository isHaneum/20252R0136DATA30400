import torch
import torch.nn as nn
import torch.nn.functional as F
    
#라벨 간의 관계를 학습하는 gcn 모듈
class LabelGCN(nn.Module):
    
    
    #input: 라벨 임베딩, 인접 행렬
    #output: 관계 정보 라벨 임베딩
    
    def __init__(self, in_dim, hidden_dim, out_dim, adj):
        super().__init__()
        # 인접 행렬 buffer로 등록
        self.register_buffer('adj', adj)
        
        # GCN 가중치 (Weight Matrices)
        self.gc1 = nn.Linear(in_dim, hidden_dim, bias=False)
        self.gc2 = nn.Linear(hidden_dim, out_dim, bias=False)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5) # 이거 수정

    def forward(self, x):
        # Layer 1: (Adj * x) * W1 -> ReLU -> Dropout
        # adj를 곱해 이웃 노드의 정보 가져오기
        x = torch.mm(self.adj, x)
        x = self.relu(self.gc1(x))
        x = self.dropout(x)
        
        # Layer 2: (Adj * X) * W2
        x = torch.mm(self.adj, x)
        x = self.gc2(x)
        return x



#최종 분류기 모델
class GCNClassifier(nn.Module):
    
    #1. 문서 벡터 투영
    #2. 라벨 벡터 refine
    #3. Dot Product
    
    def __init__(self, doc_dim, label_dim, adj, num_classes=916, label_init_emb=None):
        super().__init__()
        
        # 1. 문서 임베딩 투영 (768 -> GCN dimension)
        self.doc_proj = nn.Linear(doc_dim, label_dim)
        
        # 2. 라벨 임베딩 reset
        if label_init_emb is not None:
            self.label_embeddings = nn.Parameter(label_init_emb)
        else:
            # 랜덤 reset
            self.label_embeddings = nn.Parameter(torch.randn(num_classes, label_dim))
        
        # 3. 라벨 GCN 모듈 연결
        # Hidden Dimensio = 256
        self.label_gcn = LabelGCN(label_dim, 256, label_dim, adj)

    def forward(self, doc_embs):
        # A. 문서 벡터 변환
        doc_feat = self.doc_proj(doc_embs)
        
        # B. label vector 업데이트 
        # 라벨끼리 정보를 교환
        refined_label_emb = self.label_gcn(self.label_embeddings)
        
        # C. 최종 예측 (Matrix Multiplication)
        logits = torch.mm(doc_feat, refined_label_emb.t())
        
        return logits