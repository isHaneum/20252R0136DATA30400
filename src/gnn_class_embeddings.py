"""
GNN-based Class Embeddings for Taxonomy-aware Classification

이 모듈은 PyTorch Geometric을 사용하여 Taxonomy Graph에서 class embedding을 학습합니다.
GAT (Graph Attention Network)를 사용하여 계층 구조를 효과적으로 모델링합니다.

핵심 아이디어:
1. Taxonomy Graph를 구축 (Parent -> Child edges)
2. 초기 class embedding으로 BERT/SentenceTransformer embedding 사용
3. GAT를 통해 계층 구조를 반영한 refined embedding 학습
4. 학습된 embedding을 Training/Inference에서 활용

사용법:
    from gnn_class_embeddings import GNNClassEmbedder
    
    embedder = GNNClassEmbedder(cfg, d)
    embedder.train_gnn(epochs=100)
    class_embeddings = embedder.get_embeddings()
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch_geometric.nn import GATConv, GCNConv, SAGEConv, TransformerConv
    from torch_geometric.data import Data
    from torch_geometric.utils import add_self_loops, to_undirected
    TORCH_GEOMETRIC_AVAILABLE = True
except ImportError:
    TORCH_GEOMETRIC_AVAILABLE = False
    print("Warning: torch_geometric not installed. Run: pip install torch_geometric")

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False


# ============================================================================
# GNN Models
# ============================================================================

class GATClassifier(nn.Module):
    """
    Graph Attention Network for Class Embedding
    
    GAT는 이웃 노드에 대한 attention weight를 학습하여
    Taxonomy에서 중요한 관계를 자동으로 포착합니다.
    """
    
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 256,
        out_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Input projection
        self.input_proj = nn.Linear(in_dim, hidden_dim)
        
        # GAT layers
        self.gat_layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        
        for i in range(num_layers):
            if i == 0:
                self.gat_layers.append(
                    GATConv(hidden_dim, hidden_dim // num_heads, heads=num_heads, dropout=dropout, concat=True)
                )
            elif i == num_layers - 1:
                self.gat_layers.append(
                    GATConv(hidden_dim, out_dim, heads=1, dropout=dropout, concat=False)
                )
            else:
                self.gat_layers.append(
                    GATConv(hidden_dim, hidden_dim // num_heads, heads=num_heads, dropout=dropout, concat=True)
                )
            self.norms.append(nn.LayerNorm(hidden_dim if i < num_layers - 1 else out_dim))
        
        # Output projection
        self.output_proj = nn.Linear(out_dim, out_dim)
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        # Input projection
        x = self.input_proj(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # GAT layers with residual connections
        for i, (gat, norm) in enumerate(zip(self.gat_layers, self.norms)):
            identity = x if i < self.num_layers - 1 else None
            x = gat(x, edge_index)
            x = norm(x)
            if identity is not None and identity.shape == x.shape:
                x = x + identity  # Residual connection
            if i < self.num_layers - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Output projection
        x = self.output_proj(x)
        x = F.normalize(x, p=2, dim=-1)  # L2 normalize
        
        return x


class HierarchicalGNN(nn.Module):
    """
    Hierarchical GNN that explicitly models parent-child relationships.
    
    이 모델은 Taxonomy의 계층 구조를 명시적으로 모델링합니다:
    1. Bottom-up: Child → Parent 메시지 전달 (일반화)
    2. Top-down: Parent → Child 메시지 전달 (특화)
    """
    
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 256,
        out_dim: int = 128,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.dropout = dropout
        
        # Input projection
        self.input_proj = nn.Linear(in_dim, hidden_dim)
        
        # Bottom-up (child -> parent)
        self.bottom_up = GATConv(hidden_dim, hidden_dim // num_heads, heads=num_heads, dropout=dropout)
        self.bu_norm = nn.LayerNorm(hidden_dim)
        
        # Top-down (parent -> child)
        self.top_down = GATConv(hidden_dim, hidden_dim // num_heads, heads=num_heads, dropout=dropout)
        self.td_norm = nn.LayerNorm(hidden_dim)
        
        # Final fusion
        self.fusion = nn.Linear(hidden_dim * 2, out_dim)
        self.output_norm = nn.LayerNorm(out_dim)
        
    def forward(
        self,
        x: torch.Tensor,
        edge_index_bu: torch.Tensor,  # child -> parent edges
        edge_index_td: torch.Tensor,  # parent -> child edges
    ) -> torch.Tensor:
        # Input projection
        x = self.input_proj(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Bottom-up pass
        x_bu = self.bottom_up(x, edge_index_bu)
        x_bu = self.bu_norm(x_bu)
        x_bu = F.relu(x_bu)
        
        # Top-down pass
        x_td = self.top_down(x, edge_index_td)
        x_td = self.td_norm(x_td)
        x_td = F.relu(x_td)
        
        # Fusion
        x_fused = torch.cat([x_bu, x_td], dim=-1)
        x_out = self.fusion(x_fused)
        x_out = self.output_norm(x_out)
        x_out = F.normalize(x_out, p=2, dim=-1)
        
        return x_out


class GraphTransformer(nn.Module):
    """
    Graph Transformer for Class Embedding
    
    Transformer 구조를 그래프에 적용하여 더 풍부한 표현을 학습합니다.
    """
    
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 256,
        out_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Input projection
        self.input_proj = nn.Linear(in_dim, hidden_dim)
        
        # Transformer layers
        self.transformer_layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        
        for i in range(num_layers):
            out_channels = out_dim if i == num_layers - 1 else hidden_dim
            self.transformer_layers.append(
                TransformerConv(hidden_dim if i == 0 else hidden_dim, out_channels // num_heads, heads=num_heads, dropout=dropout, concat=True if i < num_layers - 1 else False)
            )
            self.norms.append(nn.LayerNorm(out_channels if i == num_layers - 1 else hidden_dim))
        
        self.output_proj = nn.Linear(out_dim if num_layers > 0 else hidden_dim, out_dim)
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        for i, (layer, norm) in enumerate(zip(self.transformer_layers, self.norms)):
            x = layer(x, edge_index)
            x = norm(x)
            if i < self.num_layers - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = self.output_proj(x)
        x = F.normalize(x, p=2, dim=-1)
        
        return x


# ============================================================================
# GNN Class Embedder
# ============================================================================

@dataclass
class GNNConfig:
    """GNN 설정"""
    model_type: str = "gat"  # gat | hierarchical | transformer
    hidden_dim: int = 256
    out_dim: int = 128
    num_heads: int = 4
    num_layers: int = 2
    dropout: float = 0.1
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    
    # Initial embedding
    embed_model: str = "BAAI/bge-m3"
    embed_batch_size: int = 64


class GNNClassEmbedder:
    """
    GNN-based Class Embedder
    
    Taxonomy Graph에서 class embedding을 학습합니다.
    """
    
    def __init__(
        self,
        id2name: Dict[str, str],
        keywords: Dict[str, List[str]],
        child2parents: Dict[str, List[str]],
        parent2children: Dict[str, List[str]],
        config: Optional[GNNConfig] = None,
        device: Optional[str] = None,
    ):
        if not TORCH_GEOMETRIC_AVAILABLE:
            raise ImportError("torch_geometric is required. Install with: pip install torch_geometric")
        
        self.id2name = id2name
        self.keywords = keywords
        self.child2parents = child2parents
        self.parent2children = parent2children
        self.config = config or GNNConfig()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Build class ID mapping
        self.class_ids = sorted(id2name.keys())
        self.id2idx = {cid: i for i, cid in enumerate(self.class_ids)}
        self.idx2id = {i: cid for cid, i in self.id2idx.items()}
        self.num_classes = len(self.class_ids)
        
        # Placeholders
        self.initial_embeddings: Optional[torch.Tensor] = None
        self.graph_data: Optional[Data] = None
        self.model: Optional[nn.Module] = None
        self.trained_embeddings: Optional[torch.Tensor] = None
        
        logging.info(f"GNNClassEmbedder initialized with {self.num_classes} classes")
    
    def _build_class_texts(self) -> List[str]:
        """Build text representations for each class."""
        texts = []
        for cid in self.class_ids:
            name = self.id2name.get(cid, "").replace("_", " ")
            kw = self.keywords.get(cid, [])
            texts.append(" ".join([name] + kw))
        return texts
    
    def _compute_initial_embeddings(self) -> torch.Tensor:
        """Compute initial embeddings using SentenceTransformer."""
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError("sentence_transformers is required")
        
        logging.info(f"Computing initial embeddings with {self.config.embed_model}...")
        
        model = SentenceTransformer(self.config.embed_model, device=self.device)
        texts = self._build_class_texts()
        
        embeddings = model.encode(
            texts,
            batch_size=self.config.embed_batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,
        )
        
        return torch.tensor(embeddings, dtype=torch.float32)
    
    def _build_graph(self) -> Data:
        """Build PyG Data object from taxonomy."""
        logging.info("Building taxonomy graph...")
        
        # Build edge list (parent -> child, bidirectional for message passing)
        edges_src = []
        edges_dst = []
        
        for child, parents in self.child2parents.items():
            if child not in self.id2idx:
                continue
            child_idx = self.id2idx[child]
            for parent in parents:
                if parent not in self.id2idx:
                    continue
                parent_idx = self.id2idx[parent]
                # Parent -> Child
                edges_src.append(parent_idx)
                edges_dst.append(child_idx)
                # Child -> Parent (for bidirectional)
                edges_src.append(child_idx)
                edges_dst.append(parent_idx)
        
        edge_index = torch.tensor([edges_src, edges_dst], dtype=torch.long)
        
        # Add self-loops
        edge_index, _ = add_self_loops(edge_index, num_nodes=self.num_classes)
        
        data = Data(
            x=self.initial_embeddings,
            edge_index=edge_index,
            num_nodes=self.num_classes,
        )
        
        logging.info(f"Graph: {data.num_nodes} nodes, {data.num_edges} edges")
        
        return data
    
    def _build_hierarchical_edges(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Build separate edge tensors for bottom-up and top-down."""
        bu_src, bu_dst = [], []  # child -> parent
        td_src, td_dst = [], []  # parent -> child
        
        for child, parents in self.child2parents.items():
            if child not in self.id2idx:
                continue
            child_idx = self.id2idx[child]
            for parent in parents:
                if parent not in self.id2idx:
                    continue
                parent_idx = self.id2idx[parent]
                bu_src.append(child_idx)
                bu_dst.append(parent_idx)
                td_src.append(parent_idx)
                td_dst.append(child_idx)
        
        edge_index_bu = torch.tensor([bu_src, bu_dst], dtype=torch.long)
        edge_index_td = torch.tensor([td_src, td_dst], dtype=torch.long)
        
        # Add self-loops
        edge_index_bu, _ = add_self_loops(edge_index_bu, num_nodes=self.num_classes)
        edge_index_td, _ = add_self_loops(edge_index_td, num_nodes=self.num_classes)
        
        return edge_index_bu, edge_index_td
    
    def _create_model(self, in_dim: int) -> nn.Module:
        """Create GNN model based on config."""
        if self.config.model_type == "gat":
            return GATClassifier(
                in_dim=in_dim,
                hidden_dim=self.config.hidden_dim,
                out_dim=self.config.out_dim,
                num_heads=self.config.num_heads,
                num_layers=self.config.num_layers,
                dropout=self.config.dropout,
            )
        elif self.config.model_type == "hierarchical":
            return HierarchicalGNN(
                in_dim=in_dim,
                hidden_dim=self.config.hidden_dim,
                out_dim=self.config.out_dim,
                num_heads=self.config.num_heads,
                dropout=self.config.dropout,
            )
        elif self.config.model_type == "transformer":
            return GraphTransformer(
                in_dim=in_dim,
                hidden_dim=self.config.hidden_dim,
                out_dim=self.config.out_dim,
                num_heads=self.config.num_heads,
                num_layers=self.config.num_layers,
                dropout=self.config.dropout,
            )
        else:
            raise ValueError(f"Unknown model type: {self.config.model_type}")
    
    def prepare(self) -> None:
        """Prepare embeddings and graph."""
        # Compute initial embeddings
        self.initial_embeddings = self._compute_initial_embeddings()
        
        # Build graph
        self.graph_data = self._build_graph()
        
        # Create model
        in_dim = self.initial_embeddings.shape[1]
        self.model = self._create_model(in_dim)
        self.model.to(self.device)
        
        logging.info(f"Model created: {self.config.model_type}")
    
    def train_gnn(
        self,
        epochs: int = 100,
        use_contrastive: bool = True,
        margin: float = 0.5,
        patience: int = 10,
    ) -> Dict[str, List[float]]:
        """
        Train GNN with contrastive learning on taxonomy structure.
        
        학습 목표:
        1. 같은 부모를 가진 sibling 노드들은 가까워야 함
        2. Parent-Child 관계는 가까워야 함
        3. 관련 없는 노드들은 멀어야 함
        """
        if self.model is None:
            self.prepare()
        
        self.model.train()
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        
        # Move data to device
        x = self.initial_embeddings.to(self.device)
        edge_index = self.graph_data.edge_index.to(self.device)
        
        # For hierarchical model
        if self.config.model_type == "hierarchical":
            edge_index_bu, edge_index_td = self._build_hierarchical_edges()
            edge_index_bu = edge_index_bu.to(self.device)
            edge_index_td = edge_index_td.to(self.device)
        
        # Build positive pairs (parent-child, siblings)
        positive_pairs = self._build_positive_pairs()
        negative_pairs = self._build_negative_pairs(num_negatives=len(positive_pairs) * 2)
        
        history = {"loss": [], "pos_sim": [], "neg_sim": []}
        best_loss = float("inf")
        patience_counter = 0
        
        logging.info(f"Training GNN for {epochs} epochs...")
        logging.info(f"Positive pairs: {len(positive_pairs)}, Negative pairs: {len(negative_pairs)}")
        
        for epoch in range(epochs):
            optimizer.zero_grad()
            
            # Forward pass
            if self.config.model_type == "hierarchical":
                embeddings = self.model(x, edge_index_bu, edge_index_td)
            else:
                embeddings = self.model(x, edge_index)
            
            # Contrastive loss
            loss, pos_sim, neg_sim = self._contrastive_loss(
                embeddings, positive_pairs, negative_pairs, margin
            )
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            history["loss"].append(loss.item())
            history["pos_sim"].append(pos_sim)
            history["neg_sim"].append(neg_sim)
            
            if (epoch + 1) % 10 == 0:
                logging.info(
                    f"Epoch {epoch+1}/{epochs}: loss={loss.item():.4f}, "
                    f"pos_sim={pos_sim:.4f}, neg_sim={neg_sim:.4f}"
                )
            
            # Early stopping
            if loss.item() < best_loss:
                best_loss = loss.item()
                patience_counter = 0
                # Save best embeddings
                self.trained_embeddings = embeddings.detach().cpu()
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logging.info(f"Early stopping at epoch {epoch+1}")
                    break
        
        # Final embeddings
        self.model.eval()
        with torch.no_grad():
            if self.config.model_type == "hierarchical":
                self.trained_embeddings = self.model(x, edge_index_bu, edge_index_td).cpu()
            else:
                self.trained_embeddings = self.model(x, edge_index).cpu()
        
        logging.info("GNN training complete!")
        
        return history
    
    def _build_positive_pairs(self) -> List[Tuple[int, int]]:
        """Build positive pairs (parent-child, siblings)."""
        pairs = []
        
        # Parent-child pairs
        for child, parents in self.child2parents.items():
            if child not in self.id2idx:
                continue
            child_idx = self.id2idx[child]
            for parent in parents:
                if parent not in self.id2idx:
                    continue
                parent_idx = self.id2idx[parent]
                pairs.append((child_idx, parent_idx))
        
        # Sibling pairs (same parent)
        for parent, children in self.parent2children.items():
            children_in_graph = [c for c in children if c in self.id2idx]
            for i in range(len(children_in_graph)):
                for j in range(i + 1, len(children_in_graph)):
                    pairs.append((self.id2idx[children_in_graph[i]], self.id2idx[children_in_graph[j]]))
        
        return pairs
    
    def _build_negative_pairs(self, num_negatives: int) -> List[Tuple[int, int]]:
        """Build negative pairs (random unrelated nodes)."""
        import random
        
        # Find all connected pairs
        connected = set()
        for child, parents in self.child2parents.items():
            if child not in self.id2idx:
                continue
            child_idx = self.id2idx[child]
            for parent in parents:
                if parent not in self.id2idx:
                    continue
                parent_idx = self.id2idx[parent]
                connected.add((min(child_idx, parent_idx), max(child_idx, parent_idx)))
        
        # Sample negative pairs
        pairs = []
        attempts = 0
        while len(pairs) < num_negatives and attempts < num_negatives * 10:
            i = random.randint(0, self.num_classes - 1)
            j = random.randint(0, self.num_classes - 1)
            if i != j and (min(i, j), max(i, j)) not in connected:
                pairs.append((i, j))
            attempts += 1
        
        return pairs
    
    def _contrastive_loss(
        self,
        embeddings: torch.Tensor,
        positive_pairs: List[Tuple[int, int]],
        negative_pairs: List[Tuple[int, int]],
        margin: float,
    ) -> Tuple[torch.Tensor, float, float]:
        """Compute contrastive loss."""
        if not positive_pairs:
            return torch.tensor(0.0, device=self.device), 0.0, 0.0
        
        # Positive similarities
        pos_i = torch.tensor([p[0] for p in positive_pairs], device=self.device)
        pos_j = torch.tensor([p[1] for p in positive_pairs], device=self.device)
        pos_sim = F.cosine_similarity(embeddings[pos_i], embeddings[pos_j])
        pos_loss = (1 - pos_sim).mean()
        
        # Negative similarities
        neg_loss = torch.tensor(0.0, device=self.device)
        neg_sim_avg = 0.0
        if negative_pairs:
            neg_i = torch.tensor([p[0] for p in negative_pairs], device=self.device)
            neg_j = torch.tensor([p[1] for p in negative_pairs], device=self.device)
            neg_sim = F.cosine_similarity(embeddings[neg_i], embeddings[neg_j])
            neg_loss = F.relu(neg_sim - margin + 1).mean()  # Push below margin
            neg_sim_avg = neg_sim.mean().item()
        
        total_loss = pos_loss + neg_loss
        
        return total_loss, pos_sim.mean().item(), neg_sim_avg
    
    def get_embeddings(self) -> np.ndarray:
        """Get trained GNN embeddings."""
        if self.trained_embeddings is None:
            if self.initial_embeddings is None:
                self.prepare()
            return self.initial_embeddings.numpy()
        return self.trained_embeddings.numpy()
    
    def get_embedding_dict(self) -> Dict[str, np.ndarray]:
        """Get embeddings as a dictionary {class_id: embedding}."""
        embeddings = self.get_embeddings()
        return {cid: embeddings[self.id2idx[cid]] for cid in self.class_ids}
    
    def save(self, path: str) -> None:
        """Save trained embeddings and model."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        
        save_dict = {
            "class_ids": self.class_ids,
            "embeddings": self.get_embeddings(),
            "config": {
                "model_type": self.config.model_type,
                "hidden_dim": self.config.hidden_dim,
                "out_dim": self.config.out_dim,
                "num_heads": self.config.num_heads,
                "num_layers": self.config.num_layers,
            },
        }
        
        np.savez(path, **save_dict)
        logging.info(f"Saved GNN embeddings to {path}")
    
    def load(self, path: str) -> None:
        """Load trained embeddings."""
        data = np.load(path, allow_pickle=True)
        
        loaded_ids = list(data["class_ids"])
        loaded_emb = data["embeddings"]
        
        # Verify class IDs match
        if loaded_ids != self.class_ids:
            logging.warning("Loaded class IDs do not match current class IDs!")
        
        self.trained_embeddings = torch.tensor(loaded_emb, dtype=torch.float32)
        logging.info(f"Loaded GNN embeddings from {path}")


# ============================================================================
# Convenience Functions
# ============================================================================

def create_gnn_embedder_from_config(cfg, d) -> GNNClassEmbedder:
    """Create GNNClassEmbedder from config and parsed data."""
    gnn_config = GNNConfig()
    
    return GNNClassEmbedder(
        id2name=d.id2name,
        keywords=d.keywords,
        child2parents=d.child2parents,
        parent2children=d.parent2children,
        config=gnn_config,
    )


def main():
    """Test GNN embedder."""
    import argparse
    from config import load_config
    from data import parse_all
    
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-type", type=str, default="gat", choices=["gat", "hierarchical", "transformer"])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--out-dim", type=int, default=128)
    parser.add_argument("--save-path", type=str, default="artifacts/gnn_class_embeddings.npz")
    args = parser.parse_args()
    
    cfg = load_config()
    d = parse_all(cfg.paths)
    
    config = GNNConfig(
        model_type=args.model_type,
        hidden_dim=args.hidden_dim,
        out_dim=args.out_dim,
    )
    
    embedder = GNNClassEmbedder(
        id2name=d.id2name,
        keywords=d.keywords,
        child2parents=d.child2parents,
        parent2children=d.parent2children,
        config=config,
    )
    
    embedder.prepare()
    history = embedder.train_gnn(epochs=args.epochs)
    embedder.save(args.save_path)
    
    print(f"\nFinal embeddings shape: {embedder.get_embeddings().shape}")
    print(f"Saved to: {args.save_path}")


if __name__ == "__main__":
    main()
