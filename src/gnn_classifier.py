"""
GNN-Enhanced Multi-Label Classifier

Proper GNN Usage:
1. Text Encoder (DeBERTa) → 768-dim text embedding
2. Standard Classification Head → 31 class logits
3. GNN Layer (2-layer, Skip Connection) → Label relationship modeling

Key Idea:
- GNN **refines output logits** (not classifier weights!)
- Initial prediction → GNN for label relationship → Final prediction
- Skip connection preserves original prediction

Usage:
    python src/gnn_classifier.py
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup
from tqdm.auto import tqdm
from sklearn.metrics import f1_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import load_config
from data import parse_all
from utils import ensure_dir, read_jsonl, set_seed


# ============================================================================
# GNN Layer for Label Refinement
# ============================================================================

class LabelGNNLayer(nn.Module):
    """
    2-Layer GNN with Skip Connection for Label Refinement
    
    Input: Initial logits (batch, num_labels)
    Output: Refined logits (batch, num_labels)
    
    Refines predictions by incorporating label relationships (taxonomy).
    """
    
    def __init__(
        self,
        num_labels: int,
        hidden_dim: int = 256,
        edge_index: Optional[torch.Tensor] = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.num_labels = num_labels
        self.hidden_dim = hidden_dim
        
        # Edge index (label graph)
        if edge_index is not None:
            self.register_buffer('edge_index', edge_index)
        else:
            # Default: no edges (identity)
            self.register_buffer('edge_index', torch.zeros(2, 0, dtype=torch.long))
        
        # Layer 1: logit → hidden
        self.fc1 = nn.Linear(1, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        
        # Message passing weights
        self.msg_weight = nn.Linear(hidden_dim, hidden_dim)
        self.update_weight = nn.Linear(hidden_dim * 2, hidden_dim)
        
        # Layer 2: hidden → hidden (with skip)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
        # Output: hidden → logit
        self.fc_out = nn.Linear(hidden_dim, 1)
        
        # Skip connection weight
        self.skip_weight = nn.Parameter(torch.tensor(0.5))
        
        self.dropout = nn.Dropout(dropout)
        
    def _aggregate_neighbors(self, x: torch.Tensor) -> torch.Tensor:
        """
        Aggregate neighbor node information
        x: (batch, num_labels, hidden_dim)
        """
        batch_size = x.size(0)
        
        if self.edge_index.size(1) == 0:
            # No edges → return zeros
            return torch.zeros_like(x)
        
        src, dst = self.edge_index[0], self.edge_index[1]
        
        # Gather source features
        src_features = x[:, src, :]  # (batch, num_edges, hidden)
        
        # Aggregate to destination (mean)
        agg = torch.zeros_like(x)
        agg.scatter_add_(1, dst.unsqueeze(0).unsqueeze(-1).expand(batch_size, -1, x.size(-1)), src_features)
        
        # Normalize by degree
        degree = torch.zeros(self.num_labels, device=x.device)
        degree.scatter_add_(0, dst, torch.ones_like(dst, dtype=torch.float))
        degree = degree.clamp(min=1).unsqueeze(0).unsqueeze(-1)
        agg = agg / degree
        
        return agg
        
    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """
        logits: (batch, num_labels)
        returns: refined logits (batch, num_labels)
        """
        batch_size = logits.size(0)
        
        # Save original for skip connection
        original_logits = logits
        
        # Reshape: (batch, num_labels) → (batch, num_labels, 1)
        x = logits.unsqueeze(-1)
        
        # Layer 1
        h = self.fc1(x)  # (batch, num_labels, hidden)
        h = self.norm1(h)
        h = F.gelu(h)
        h = self.dropout(h)
        
        # Message passing (if edges exist)
        if self.edge_index.size(1) > 0:
            msg = self._aggregate_neighbors(self.msg_weight(h))
            h = self.update_weight(torch.cat([h, msg], dim=-1))
            h = F.gelu(h)
            h = self.dropout(h)
        
        # Layer 2 with skip connection
        h2 = self.fc2(h)
        h2 = self.norm2(h2)
        h2 = h2 + h  # Skip connection within GNN
        h2 = F.gelu(h2)
        h2 = self.dropout(h2)
        
        # Output
        refined = self.fc_out(h2).squeeze(-1)  # (batch, num_labels)
        
        # Final skip connection: blend original and refined
        alpha = torch.sigmoid(self.skip_weight)
        output = alpha * refined + (1 - alpha) * original_logits
        
        return output


# ============================================================================
# Main Classifier
# ============================================================================

class GNNMultiLabelClassifier(nn.Module):
    """
    GNN-Enhanced Multi-Label Classifier
    
    Architecture:
    1. Text Encoder (DeBERTa) → CLS embedding (768)
    2. Classification Head → Initial logits (num_labels)
    3. Label GNN (2-layer, skip connection) → Refined logits (num_labels)
    """
    
    def __init__(
        self,
        encoder_name: str,
        num_labels: int,
        edge_index: Optional[torch.Tensor] = None,
        gnn_hidden_dim: int = 256,
        use_gnn: bool = True,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        # Text encoder
        self.encoder = AutoModel.from_pretrained(encoder_name)
        self.encoder_dim = self.encoder.config.hidden_size
        self.num_labels = num_labels
        self.use_gnn = use_gnn
        
        # Classification head (standard)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.encoder_dim, self.encoder_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.encoder_dim, num_labels),
        )
        
        # Label GNN for refinement
        if use_gnn:
            self.label_gnn = LabelGNNLayer(
                num_labels=num_labels,
                hidden_dim=gnn_hidden_dim,
                edge_index=edge_index,
                dropout=dropout,
            )
        
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        # Text encoding
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        text_emb = outputs.last_hidden_state[:, 0, :]  # CLS token
        
        # Initial classification
        logits = self.classifier(text_emb)
        
        # GNN refinement
        if self.use_gnn and hasattr(self, 'label_gnn'):
            logits = self.label_gnn(logits)
        
        return logits


# ============================================================================
# Dataset
# ============================================================================

class MultiLabelDataset(Dataset):
    def __init__(self, rows: List[dict], label2id: Dict[str, int], num_labels: int):
        self.rows = rows
        self.label2id = label2id
        self.num_labels = num_labels
    
    def __len__(self) -> int:
        return len(self.rows)
    
    def __getitem__(self, idx: int):
        r = self.rows[idx]
        y = np.zeros(self.num_labels, dtype=np.float32)
        for lab in r['labels']:
            if str(lab) in self.label2id:
                y[self.label2id[str(lab)]] = 1.0
        return {
            'text': r['text'],
            'labels': torch.tensor(y, dtype=torch.float32),
        }


def make_collate(tokenizer, max_length: int):
    def collate(batch):
        texts = [x['text'] for x in batch]
        enc = tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors='pt',
        )
        enc['labels'] = torch.stack([x['labels'] for x in batch])
        return enc
    return collate


# ============================================================================
# Loss with Positive Weighting
# ============================================================================

class FocalLoss(nn.Module):
    """Focal Loss for multi-label classification"""
    def __init__(self, gamma: float = 2.0, pos_weight: float = 10.0):
        super().__init__()
        self.gamma = gamma
        self.pos_weight = pos_weight
        
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        
        # Focal weight
        pt = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = (1 - pt) ** self.gamma
        
        # Positive weight
        weight = torch.where(targets == 1, self.pos_weight, 1.0)
        
        # BCE
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        
        loss = focal_weight * weight * bce
        return loss.mean()


# ============================================================================
# Training
# ============================================================================

def build_edge_index(child2parents: Dict[str, List[str]], label2id: Dict[str, int]) -> torch.Tensor:
    """Build edge index from taxonomy structure"""
    src, dst = [], []
    
    for child, parents in child2parents.items():
        if str(child) not in label2id:
            continue
        child_idx = label2id[str(child)]
        for parent in parents:
            if str(parent) not in label2id:
                continue
            parent_idx = label2id[str(parent)]
            # Bidirectional edges
            src.extend([child_idx, parent_idx])
            dst.extend([parent_idx, child_idx])
    
    if not src:
        return torch.zeros(2, 0, dtype=torch.long)
    
    return torch.tensor([src, dst], dtype=torch.long)


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler,
    device: torch.device,
    criterion: nn.Module,
) -> float:
    model.train()
    total_loss = 0.0
    
    pbar = tqdm(dataloader, desc="Training", leave=False)
    for batch in pbar:
        optimizer.zero_grad()
        
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        with torch.amp.autocast('cuda'):
            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(logits, labels)
        
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        
        total_loss += loss.item()
        pbar.set_postfix(loss=loss.item())
    
    return total_loss / len(dataloader)


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    top_k: int = 3,
) -> Dict[str, float]:
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].numpy()
            
            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.sigmoid(logits).cpu().numpy()
            
            # Top-K prediction
            preds = np.zeros_like(probs, dtype=np.int32)
            for i in range(probs.shape[0]):
                top_indices = np.argsort(probs[i])[-top_k:]
                preds[i, top_indices] = 1
            
            all_preds.append(preds)
            all_labels.append(labels)
            all_probs.append(probs)
    
    all_preds = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)
    all_probs = np.vstack(all_probs)
    
    # Debug
    print(f"[DEBUG] probs max: {all_probs.max():.4f}, min: {all_probs.min():.4f}, mean: {all_probs.mean():.4f}")
    print(f"[DEBUG] probs std: {all_probs.std():.4f}")
    print(f"[DEBUG] top-{top_k} preds: {np.sum(all_preds)}, labels>0: {np.sum(all_labels)}")
    
    micro_f1 = f1_score(all_labels, all_preds, average='micro', zero_division=0)
    samples_f1 = f1_score(all_labels, all_preds, average='samples', zero_division=0)
    
    return {'micro_f1': micro_f1, 'samples_f1': samples_f1}


# ============================================================================
# Main
# ============================================================================

def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder", type=str, default="microsoft/deberta-v3-base")
    parser.add_argument("--use-gnn", action="store_true", default=True)
    parser.add_argument("--gnn-hidden", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=192)
    parser.add_argument("--val-ratio", type=float, default=0.05)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--save-dir", type=str, default="student_gnn")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    set_seed(args.seed)
    
    cfg = load_config()
    d = parse_all(cfg.paths)
    
    # Load silver data
    silver_file = cfg.paths.silver_file
    logging.info(f"Loading silver data from {silver_file}")
    silver_data = read_jsonl(silver_file)
    logging.info(f"Loaded {len(silver_data)} samples")
    
    # Build label mapping
    class_ids = sorted(d.id2name.keys())
    label2id = {str(cid): i for i, cid in enumerate(class_ids)}
    num_labels = len(class_ids)
    logging.info(f"Number of labels: {num_labels}")
    
    # Build edge index from taxonomy
    edge_index = build_edge_index(d.child2parents, label2id)
    logging.info(f"Edge index: {edge_index.shape[1]} edges")
    
    # Split train/val
    np.random.shuffle(silver_data)
    val_size = int(len(silver_data) * args.val_ratio)
    train_data = silver_data[val_size:]
    val_data = silver_data[:val_size]
    logging.info(f"Train: {len(train_data)}, Val: {len(val_data)}")
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Device: {device}")
    
    # Model
    model = GNNMultiLabelClassifier(
        encoder_name=args.encoder,
        num_labels=num_labels,
        edge_index=edge_index.to(device) if edge_index.size(1) > 0 else None,
        gnn_hidden_dim=args.gnn_hidden,
        use_gnn=args.use_gnn,
    )
    model.to(device)
    logging.info(f"Model created, use_gnn={args.use_gnn}")
    
    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.encoder, use_fast=False)
    
    # Datasets
    train_ds = MultiLabelDataset(train_data, label2id, num_labels)
    val_ds = MultiLabelDataset(val_data, label2id, num_labels)
    
    train_dl = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        collate_fn=make_collate(tokenizer, args.max_length),
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=make_collate(tokenizer, args.max_length),
    )
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = len(train_dl) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * 0.1),
        num_training_steps=total_steps,
    )
    scaler = torch.amp.GradScaler('cuda')
    
    # Loss (Focal Loss with positive weighting)
    criterion = FocalLoss(gamma=2.0, pos_weight=10.0)
    
    # Training loop
    best_f1 = 0.0
    patience_counter = 0
    
    for epoch in range(args.epochs):
        logging.info(f"\n=== Epoch {epoch+1}/{args.epochs} ===")
        
        train_loss = train_epoch(model, train_dl, optimizer, scheduler, scaler, device, criterion)
        logging.info(f"Train loss: {train_loss:.4f}")
        
        metrics = evaluate(model, val_dl, device)
        logging.info(f"Val micro_f1: {metrics['micro_f1']:.4f}, samples_f1: {metrics['samples_f1']:.4f}")
        
        if metrics['micro_f1'] > best_f1:
            best_f1 = metrics['micro_f1']
            patience_counter = 0
            
            # Save model
            save_dir = args.save_dir
            ensure_dir(save_dir)
            model.encoder.save_pretrained(save_dir)
            tokenizer.save_pretrained(save_dir)
            
            with open(os.path.join(save_dir, "label2id.json"), "w") as f:
                json.dump(label2id, f)
            
            torch.save(model.state_dict(), os.path.join(save_dir, "model_state.pt"))
            
            # Save edge index
            torch.save(edge_index, os.path.join(save_dir, "edge_index.pt"))
            
            logging.info(f"Saved best model with micro_f1={best_f1:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                logging.info(f"Early stopping at epoch {epoch+1}")
                break
    
    logging.info(f"\nTraining complete. Best micro_f1: {best_f1:.4f}")


if __name__ == "__main__":
    main()
