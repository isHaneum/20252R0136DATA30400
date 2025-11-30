import torch
import numpy as np
import config
import os

def build_adjacency_matrix(num_classes=None):
    print(f"Building Taxonomy Graph from {config.TAXONOMY_PATH}...")
    
    # 1. 족보 파일 읽기
    edges = []
    max_id = 0
    
    if os.path.exists(config.TAXONOMY_PATH):
        with open(config.TAXONOMY_PATH, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    try:
                        # 부모-자식 관계 (ParentID ChildID)
                        p, c = int(parts[0]), int(parts[1])
                        edges.append((p, c))
                        max_id = max(max_id, p, c)
                    except ValueError:
                        continue
    else:
        print(f"Taxonomy file not found: {config.TAXONOMY_PATH}")
        return None

    # 클래스 개수 설정 (Config 값 우선, 없으면 파일에서 발견된 최대 ID + 1)
    if num_classes is None:
        num_classes = max_id + 1
    
    print(f"Graph Nodes: {num_classes}, Edges: {len(edges)}")

    # 2. 인접 행렬 생성 (Identity Matrix로 초기화 -> Self-loop 포함)
    adj = torch.eye(num_classes)

    for p, c in edges:
        if p < num_classes and c < num_classes:
            adj[p, c] = 1
            adj[c, p] = 1  # 양방향 연결 (Undirected)

    # 3. 정규화 (Normalization: D^-0.5 * A * D^-0.5)
    # 차수(Degree) 계산: 각 노드가 몇 개랑 연결됐는지
    row_sum = torch.sum(adj, dim=1)
    
    # 역제곱근 계산 (Division by zero 방지)
    d_inv_sqrt = torch.pow(row_sum, -0.5)
    d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.
    
    # 대각 행렬로 변환
    d_mat_inv_sqrt = torch.diag(d_inv_sqrt)
    
    # 행렬 곱: D^-0.5 * A * D^-0.5
    norm_adj = torch.mm(torch.mm(d_mat_inv_sqrt, adj), d_mat_inv_sqrt)
    
    return norm_adj

if __name__ == "__main__":
    # 테스트 실행
    adj = build_adjacency_matrix()
    if adj is not None:
        print(f"✅ Adjacency Matrix Created: {adj.shape}")