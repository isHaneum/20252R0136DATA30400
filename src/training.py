<<<<<<< HEAD
from __future__ import annotations

import argparse
import json
import logging
import os
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup
from tqdm.auto import tqdm
from sklearn.metrics import f1_score

from config import load_config
from data import parse_all
from utils import ensure_dir, read_jsonl, set_seed


@dataclass
class TrainRow:
    text: str
    y: np.ndarray
    conf: float = 1.0
    input_ids: torch.Tensor | None = None
    attention_mask: torch.Tensor | None = None


class SilverDataset(Dataset):
    def __init__(self, rows: List[TrainRow]):
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        r = self.rows[idx]
        item = {
            "labels": torch.tensor(r.y, dtype=torch.float32),
            "conf": torch.tensor(float(r.conf), dtype=torch.float32),
        }
        if r.input_ids is not None:
            item["input_ids"] = r.input_ids
            item["attention_mask"] = r.attention_mask
        else:
            # Fallback for non-pretokenized (should not happen if pretokenized)
            item["text"] = r.text
        return item


def make_collate(tokenizer, max_length: int):
    def collate(batch: List[dict]):
        # Check if pre-tokenized
        if "input_ids" in batch[0]:
            input_ids = torch.stack([x["input_ids"] for x in batch])
            attention_mask = torch.stack([x["attention_mask"] for x in batch])
            labels = torch.stack([x["labels"] for x in batch])
            conf = torch.stack([x["conf"] for x in batch])
            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
                "conf": conf,
            }
        
        # Fallback to on-the-fly tokenization
        texts = [x["text"] for x in batch]
        enc = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        batch_out = {k: v for k, v in enc.items()}
        batch_out["labels"] = torch.stack([x["labels"] for x in batch])
        batch_out["conf"] = torch.stack([x["conf"] for x in batch])
        return batch_out

    return collate


def _load_silver(path: str) -> List[dict]:
    rows = read_jsonl(path)
    out: List[dict] = []
    for r in rows:
        if isinstance(r, dict) and "text" in r and "labels" in r:
            out.append(r)
    return out


def _build_label_index(class_ids: List[str]) -> Tuple[Dict[str, int], Dict[int, str]]:
    label2id = {cid: i for i, cid in enumerate(class_ids)}
    id2label = {i: cid for cid, i in label2id.items()}
    return label2id, id2label


def _train_loop(
    *,
    model,
    dl: DataLoader,
    device: torch.device,
    optim,
    scheduler,
    scaler,
    use_fp16: bool,
    grad_accum: int,
    use_confidence_weight: bool,
    max_grad_norm: float,
    epochs: int,
    desc_prefix: str,
    # early stopping
    val_dl: DataLoader | None = None,
    patience: int = 0,
    min_delta: float = 0.0,
    eval_topk: int | None = 3,
    eval_threshold: float | None = None,
    eval_topk_grid: List[int] | None = None,
    eval_threshold_grid: List[float] | None = None,
    early_metric: str = "micro",
    progress: str = "epoch",  # batch | epoch | none
    pos_weight: torch.Tensor | None = None,
) -> None:
    def _ascii_bar(x: float, *, width: int = 20) -> str:
        x = float(x)
        x = 0.0 if np.isnan(x) or np.isinf(x) else max(0.0, min(1.0, x))
        filled = int(round(x * width))
        return "[" + ("█" * filled) + ("·" * (width - filled)) + "]"

    def _evaluate_all() -> dict:
        if val_dl is None:
            return {
                "micro": float("nan"),
                "samples": float("nan"),
                "micro_best": float("nan"),
                "samples_best": float("nan"),
                "best_mode": None,
            }
        model.eval()
        ys: List[np.ndarray] = []
        probs_list: List[np.ndarray] = []
        with torch.no_grad():
            for batch in val_dl:
                _ = batch.pop("conf", None)
                batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
                labels = batch["labels"].detach().cpu().numpy().astype(np.int32)
                logits = model(**batch).logits
                probs = torch.sigmoid(logits).detach().cpu().numpy().astype(np.float32)
                ys.append(labels)
                probs_list.append(probs)

        y_all = np.concatenate(ys, axis=0)
        probs_all = np.concatenate(probs_list, axis=0)

        def _score(pred: np.ndarray) -> tuple[float, float]:
            micro = float(f1_score(y_all, pred, average="micro", zero_division=0))
            samples = float(f1_score(y_all, pred, average="samples", zero_division=0))
            return micro, samples

        # Default (for compatibility with existing logs)
        if eval_threshold is not None:
            pred_default = (probs_all >= float(eval_threshold)).astype(np.int32)
            micro, samples = _score(pred_default)
        else:
            k0 = int(eval_topk or 3)
            k0 = max(1, min(k0, probs_all.shape[1]))
            idx = np.argpartition(-probs_all, kth=k0 - 1, axis=1)[:, :k0]
            pred_default = np.zeros_like(probs_all, dtype=np.int32)
            rows = np.arange(pred_default.shape[0])[:, None]
            pred_default[rows, idx] = 1
            micro, samples = _score(pred_default)

        best_micro = micro
        best_samples = samples
        best_mode = ("threshold", float(eval_threshold)) if eval_threshold is not None else ("topk", int(eval_topk or 3))

        # Optional sweeps (cheap on small val sets, very helpful for sanity)
        if eval_topk_grid:
            for k in eval_topk_grid:
                k = int(k)
                if k <= 0:
                    continue
                k = min(k, probs_all.shape[1])
                idx = np.argpartition(-probs_all, kth=k - 1, axis=1)[:, :k]
                pred = np.zeros_like(probs_all, dtype=np.int32)
                rows = np.arange(pred.shape[0])[:, None]
                pred[rows, idx] = 1
                m, s = _score(pred)
                # Track best by early_metric
                metric_val = m if early_metric == "micro" else s
                best_metric_val = best_micro if early_metric == "micro" else best_samples
                if metric_val > best_metric_val:
                    best_micro, best_samples = m, s
                    best_mode = ("topk", k)

        if eval_threshold_grid:
            for thr in eval_threshold_grid:
                thr = float(thr)
                if not (0.0 <= thr <= 1.0):
                    continue
                pred = (probs_all >= thr).astype(np.int32)
                m, s = _score(pred)
                metric_val = m if early_metric == "micro" else s
                best_metric_val = best_micro if early_metric == "micro" else best_samples
                if metric_val > best_metric_val:
                    best_micro, best_samples = m, s
                    best_mode = ("threshold", thr)

        return {
            "micro": micro,
            "samples": samples,
            "micro_best": best_micro,
            "samples_best": best_samples,
            "best_mode": best_mode,
            "bar_micro": _ascii_bar(best_micro),
            "bar_samples": _ascii_bar(best_samples),
        }

    model.train()
    early_metric = str(early_metric)
    if early_metric not in ("micro", "samples"):
        early_metric = "micro"

    best_metric = -1e9
    bad_epochs = 0
    best_path = None

    progress = str(progress)
    show_epoch_pbar = progress in ("epoch", "batch")
    epoch_pbar = tqdm(
        range(int(epochs)),
        desc=f"{desc_prefix} epochs",
        unit="epoch",
        mininterval=10.0,
        maxinterval=10.0,
        disable=not show_epoch_pbar,
    )

    micro_hist: List[float] = []
    samples_hist: List[float] = []
    for epoch in epoch_pbar:
        total_loss = 0.0
        optim.zero_grad(set_to_none=True)
        step_in_epoch = 0
        show_batch_pbar = progress == "batch"
        pbar = tqdm(
            dl,
            desc=f"{desc_prefix} epoch {epoch + 1}/{int(epochs)}",
            unit="batch",
            leave=False,
            mininterval=10.0,
            maxinterval=10.0,
            disable=not show_batch_pbar,
        )
        for batch in pbar:
            conf = batch.pop("conf").to(device, non_blocking=True)
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            labels = batch["labels"]

            # Use new API for autocast (PyTorch 2.9+)
            autocast_ctx = torch.amp.autocast('cuda', enabled=use_fp16) if hasattr(torch, 'amp') and hasattr(torch.amp, 'autocast') else torch.cuda.amp.autocast(enabled=use_fp16)
            with autocast_ctx:
                out = model(**batch)
                logits = out.logits

                if use_confidence_weight:
                    per_elem = F.binary_cross_entropy_with_logits(
                        logits,
                        labels,
                        reduction="none",
                        pos_weight=pos_weight,
                    )
                    per_sample = per_elem.mean(dim=1)
                    raw_loss = (per_sample * conf).mean()
                else:
                    raw_loss = F.binary_cross_entropy_with_logits(
                        logits,
                        labels,
                        pos_weight=pos_weight,
                    )

                loss = raw_loss / grad_accum

            scaler.scale(loss).backward()
            step_in_epoch += 1
            step_loss = float(raw_loss.detach().cpu().item())
            total_loss += step_loss

            if step_in_epoch % grad_accum == 0:
                scaler.unscale_(optim)
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(max_grad_norm))
                scaler.step(optim)
                scaler.update()
                scheduler.step()
                optim.zero_grad(set_to_none=True)

        # flush last partial accumulation
        if step_in_epoch % grad_accum != 0:
            scaler.unscale_(optim)
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(max_grad_norm))
            scaler.step(optim)
            scaler.update()
            scheduler.step()
            optim.zero_grad(set_to_none=True)

        avg_loss = total_loss / max(1, len(dl))

        # Always evaluate when val_dl exists (even if patience==0), so F1 is always printed.
        metrics = _evaluate_all() if val_dl is not None else None
        if metrics is not None:
            micro_hist.append(float(metrics["micro_best"]))
            samples_hist.append(float(metrics["samples_best"]))
            if show_epoch_pbar:
                epoch_pbar.set_postfix(
                    loss=f"{avg_loss:.4f}",
                    micro=f"{float(metrics['micro_best']):.4f}",
                    samples=f"{float(metrics['samples_best']):.4f}",
                )
            # Stable, non-flickering per-epoch line with a filling bar
            best_mode = metrics.get("best_mode")
            best_mode_s = ""
            if isinstance(best_mode, tuple) and len(best_mode) == 2:
                kind, v = best_mode
                if kind == "topk":
                    best_mode_s = f"best=topk@{int(v)}"
                elif kind == "threshold":
                    best_mode_s = f"best=thr@{float(v):.2f}"
            tqdm.write(
                f"{desc_prefix} epoch {epoch + 1}/{int(epochs)} | loss={avg_loss:.4f} | "
                f"micro={float(metrics['micro_best']):.4f} {metrics['bar_micro']} | "
                f"samples={float(metrics['samples_best']):.4f} {metrics['bar_samples']} {best_mode_s}".strip()
            )

        # Early stopping only if enabled
        if val_dl is not None and int(patience) > 0 and metrics is not None:
            metric = float(metrics["micro_best"]) if early_metric == "micro" else float(metrics["samples_best"])
            improved = metric > (best_metric + float(min_delta))
            if improved:
                best_metric = metric
                bad_epochs = 0
                ensure_dir(os.path.join(load_config().paths.artifacts_dir, "checkpoints"))
                best_path = os.path.join(load_config().paths.artifacts_dir, "checkpoints", f"best_{desc_prefix}.pt")
                torch.save(model.state_dict(), best_path)
            else:
                bad_epochs += 1
                if bad_epochs >= int(patience):
                    tqdm.write(f"{desc_prefix} early stopping (patience={int(patience)}, metric={early_metric})")
                    break

        model.train()

    # Restore best checkpoint if early stopping was enabled
    if val_dl is not None and int(patience) > 0 and best_path and os.path.exists(best_path):
        model.load_state_dict(torch.load(best_path, map_location=device))


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    cfg = load_config()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default=cfg.train.model_name,
        help="HF model name/path (e.g., microsoft/deberta-v3-base for faster training)",
    )
    parser.add_argument("--epochs", type=int, default=cfg.train.num_epochs)
    parser.add_argument("--batch-size", type=int, default=cfg.train.batch_size)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--min-conf", type=float, default=getattr(cfg.train, 'min_confidence', 0.0),
                        help="Skip silver samples with confidence below this threshold")
    parser.add_argument("--grad-accum", type=int, default=cfg.train.grad_accum_steps)
    parser.add_argument("--no-fp16", action="store_true")
    parser.add_argument(
        "--progress",
        type=str,
        default="epoch",
        choices=["batch", "epoch", "none"],
        help="Progress display: batch=tqdm batches, epoch=tqdm epochs + stable epoch lines, none=only stable epoch lines.",
    )
    parser.add_argument(
        "--use-pos-weight",
        action="store_true",
        default=True,
        help="Use per-label pos_weight for BCE to counter extreme class imbalance.",
    )
    parser.add_argument(
        "--no-pos-weight",
        action="store_false",
        dest="use_pos_weight",
        help="Disable pos_weight.",
    )

    # Two-stage training on pseudo labels (silver)
    parser.add_argument(
        "--stage",
        type=str,
        default="single",
        choices=["single", "pretrain", "pseudo", "selftrain", "both"],
        help="single=one pass (default). both=pretrain then pseudo fine-tune.",
    )
    parser.add_argument("--pretrain-epochs", type=int, default=1)
    parser.add_argument("--pretrain-max-rows", type=int, default=8000)
    parser.add_argument("--pretrain-min-conf", type=float, default=0.25)
    parser.add_argument("--pseudo-epochs", type=int, default=None)
    parser.add_argument("--pseudo-lr", type=float, default=None)

    # Validation + early stopping (micro-F1)
    parser.add_argument("--val-ratio", type=float, default=0.05)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--min-delta", type=float, default=0.0)
    parser.add_argument(
        "--early-metric",
        type=str,
        default="micro",
        choices=["micro", "samples"],
        help="Metric to drive early stopping on silver val split.",
    )
    parser.add_argument(
        "--eval-topk",
        type=int,
        default=3,
        help="If set, predict top-k labels per sample for F1 eval (default=3).",
    )
    parser.add_argument(
        "--eval-threshold",
        type=float,
        default=None,
        help="If set, use sigmoid>=threshold for F1 eval (overrides --eval-topk).",
    )
    parser.add_argument(
        "--eval-topk-grid",
        type=str,
        default="2,3,5",
        help="Comma-separated k values to try on val and report the best (only used when val split exists).",
    )
    parser.add_argument(
        "--eval-threshold-grid",
        type=str,
        default="0.05,0.10,0.15,0.20,0.25,0.30",
        help="Comma-separated thresholds to try on val and report the best (only used when val split exists).",
    )
    args = parser.parse_args()

    set_seed(cfg.train.seed)

    d = parse_all(cfg.paths)
    class_ids = sorted(d.id2name.keys())
    label2id, id2label = _build_label_index(class_ids)

    silver = _load_silver(cfg.paths.silver_file)
    if args.max_rows is not None:
        silver = silver[: args.max_rows]

    if len(silver) == 0:
        raise RuntimeError(f"No silver data found at {cfg.paths.silver_file}. Run retrieval + silver_labeling first.")

    rows: List[TrainRow] = []
    skipped_low_conf = 0
    min_conf_filter = float(args.min_conf) if hasattr(args, 'min_conf') else float(getattr(cfg.train, 'min_confidence', 0.0))
    
    for r in silver:
        labels = [str(x) for x in r["labels"]]
        conf = float(r.get("conf", 1.0))
        
        # Skip very low confidence samples (noisy labels)
        if conf < min_conf_filter:
            skipped_low_conf += 1
            continue
            
        y = np.zeros(len(class_ids), dtype=np.float32)
        kept = 0
        for lab in labels:
            if lab in label2id:
                y[label2id[lab]] = 1.0
                kept += 1
        if kept >= 1:
            # conf can be cosine similarity (-1..1) or reranker score; keep a safe [0.1..1.0] weight
            if np.isnan(conf) or np.isinf(conf):
                conf = 1.0
            conf = max(0.1, min(1.0, conf))
            rows.append(TrainRow(text=r["text"], y=y, conf=conf))

    logging.info("Training rows: %d (skipped %d low-conf samples)", len(rows), skipped_low_conf)

    # Always-on local smoke check on ~50 rows (fast, catches silent format bugs).
    smoke_n = min(50, len(rows))
    if smoke_n > 0:
        y_counts = [int(r.y.sum()) for r in rows[:smoke_n]]
        bad_text = sum(1 for r in rows[:smoke_n] if not str(r.text).strip())
        bad_y = sum(1 for c in y_counts if c <= 0)
        bad_conf = sum(1 for r in rows[:smoke_n] if not (0.0 <= float(r.conf) <= 1.0))
        logging.info(
            "smoke_train@%d: empty_text=%d zero_labels=%d bad_conf=%d label_count(min/mean/max)=%d/%.2f/%d",
            smoke_n,
            bad_text,
            bad_y,
            bad_conf,
            min(y_counts) if y_counts else 0,
            float(np.mean(y_counts)) if y_counts else 0.0,
            max(y_counts) if y_counts else 0,
        )

    if len(rows) < 200:
        logging.warning("Very small training set (%d rows). F1 will be unstable.", len(rows))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        # Use new API for TF32 settings (PyTorch 2.9+)
        if hasattr(torch.backends.cuda.matmul, 'fp32_precision'):
            torch.backends.cuda.matmul.fp32_precision = 'tf32'
        else:
            torch.backends.cuda.matmul.allow_tf32 = True
        if hasattr(torch.backends.cudnn.conv, 'fp32_precision'):
            torch.backends.cudnn.conv.fp32_precision = 'tf32'
        else:
            torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    model_name = str(args.model)
    # DeBERTa-v3 uses a SentencePiece tokenizer; forcing slow avoids a known fast-conversion edge case.
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(class_ids),
        problem_type="multi_label_classification",
    )
    model.to(device)

    # Compute per-label pos_weight to fight extreme imbalance (positives are ~2-3/531 per sample).
    pos_weight_t = None
    if bool(args.use_pos_weight):
        y_sum = np.zeros(len(class_ids), dtype=np.float64)
        for r in rows:
            y_sum += r.y.astype(np.float64)
        pos = y_sum
        neg = float(len(rows)) - pos
        with np.errstate(divide="ignore", invalid="ignore"):
            pw = np.where(pos > 0, neg / np.maximum(pos, 1.0), 1.0)
        # CRITICAL: Lower max clip from 50→20 to avoid over-aggressive positive prediction
        pw = np.clip(pw, 1.0, 20.0).astype(np.float32)
        pos_weight_t = torch.tensor(pw, dtype=torch.float32, device=device)
        logging.info("pos_weight stats: min=%.2f, max=%.2f, mean=%.2f", pw.min(), pw.max(), pw.mean())

    # Pre-tokenize all rows to speed up training (move CPU work to start)
    logging.info("Pre-tokenizing %d rows...", len(rows))
    all_texts = [r.text for r in rows]
    # Process in chunks to avoid OOM on very large datasets (though 30k is fine)
    chunk_size = 10000
    for i in range(0, len(rows), chunk_size):
        chunk_texts = all_texts[i : i + chunk_size]
        enc = tokenizer(
            chunk_texts,
            padding="max_length", # Pad to max length for uniform tensors
            truncation=True,
            max_length=cfg.train.max_length,
            return_tensors="pt",
        )
        for j, (input_id, attn_mask) in enumerate(zip(enc["input_ids"], enc["attention_mask"])):
            rows[i + j].input_ids = input_id
            rows[i + j].attention_mask = attn_mask
    logging.info("Pre-tokenization complete.")

    # Stage row selection
    stage = str(args.stage)
    if stage == "selftrain":
        stage = "pseudo"

    if stage in ("single", "pseudo"):
        train_rows = rows
    elif stage == "pretrain":
        train_rows = [r for r in rows if float(r.conf) >= float(args.pretrain_min_conf)]
        train_rows = train_rows[: int(args.pretrain_max_rows)]
    else:
        # both
        train_rows = rows

    # Split train/val from selected rows (for early stopping)
    val_ratio = float(args.val_ratio)
    val_dl = None
    if val_ratio > 0.0 and len(train_rows) >= 2000:
        rng = np.random.default_rng(int(cfg.train.seed))
        idx = np.arange(len(train_rows))
        rng.shuffle(idx)
        val_n = int(max(200, min(len(train_rows) * val_ratio, len(train_rows) - 1)))
        val_idx = set(idx[:val_n].tolist())
        tr = [r for i, r in enumerate(train_rows) if i not in val_idx]
        va = [r for i, r in enumerate(train_rows) if i in val_idx]
    else:
        tr = train_rows
        va = []

    ds = SilverDataset(tr)

    # Tokenization is CPU-bound; use batched tokenization in collate_fn + workers to avoid GPU starvation.
    num_workers = min(16, os.cpu_count() or 4)
    pin_memory = device.type == "cuda"
    dl_kwargs = dict(
        batch_size=int(args.batch_size),
        shuffle=True,
        num_workers=int(num_workers),
        pin_memory=bool(pin_memory),
        collate_fn=make_collate(tokenizer, cfg.train.max_length),
    )
    if int(num_workers) > 0:
        dl_kwargs.update(
            dict(
                persistent_workers=True,
                prefetch_factor=8,
            )
        )
    dl = DataLoader(ds, **dl_kwargs)
    if va:
        val_ds = SilverDataset(va)
        val_kwargs = dict(dl_kwargs)
        val_kwargs.update(dict(shuffle=False))
        val_dl = DataLoader(val_ds, **val_kwargs)

    use_fp16 = bool(cfg.train.fp16) and (not args.no_fp16) and device.type == "cuda"
    # Use new API for GradScaler (PyTorch 2.9+)
    if hasattr(torch, 'amp') and hasattr(torch.amp, 'GradScaler'):
        scaler = torch.amp.GradScaler('cuda', enabled=use_fp16)
    else:
        scaler = torch.cuda.amp.GradScaler(enabled=use_fp16)
    grad_accum = max(1, int(args.grad_accum))

    def _make_optim_and_sched(*, lr: float, steps: int):
        optim = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=cfg.train.weight_decay)
        warmup_steps = int(steps * float(cfg.train.warmup_ratio))
        sched = get_linear_schedule_with_warmup(optim, num_warmup_steps=warmup_steps, num_training_steps=int(steps))
        return optim, sched

    if stage == "single":
        total_steps = (len(dl) * int(args.epochs) + grad_accum - 1) // grad_accum
        optim, scheduler = _make_optim_and_sched(lr=cfg.train.learning_rate, steps=total_steps)
        _train_loop(
            model=model,
            dl=dl,
            device=device,
            optim=optim,
            scheduler=scheduler,
            scaler=scaler,
            use_fp16=use_fp16,
            grad_accum=grad_accum,
            use_confidence_weight=bool(cfg.train.use_confidence_weight),
            max_grad_norm=float(cfg.train.max_grad_norm),
            epochs=int(args.epochs),
            desc_prefix="train",
            val_dl=val_dl,
            patience=int(args.patience),
            min_delta=float(args.min_delta),
            eval_topk=int(args.eval_topk) if args.eval_topk is not None else None,
            eval_threshold=float(args.eval_threshold) if args.eval_threshold is not None else None,
            eval_topk_grid=[int(x) for x in str(args.eval_topk_grid).split(",") if x.strip().isdigit()],
            eval_threshold_grid=[float(x) for x in str(args.eval_threshold_grid).split(",") if x.strip()],
            early_metric=str(args.early_metric),
            progress=str(args.progress),
            pos_weight=pos_weight_t,
        )
    elif stage == "pretrain":
        total_steps = (len(dl) * int(args.pretrain_epochs) + grad_accum - 1) // grad_accum
        optim, scheduler = _make_optim_and_sched(lr=cfg.train.learning_rate, steps=total_steps)
        _train_loop(
            model=model,
            dl=dl,
            device=device,
            optim=optim,
            scheduler=scheduler,
            scaler=scaler,
            use_fp16=use_fp16,
            grad_accum=grad_accum,
            use_confidence_weight=bool(cfg.train.use_confidence_weight),
            max_grad_norm=float(cfg.train.max_grad_norm),
            epochs=int(args.pretrain_epochs),
            desc_prefix="pretrain",
            val_dl=val_dl,
            patience=int(args.patience),
            min_delta=float(args.min_delta),
            eval_topk=int(args.eval_topk) if args.eval_topk is not None else None,
            eval_threshold=float(args.eval_threshold) if args.eval_threshold is not None else None,
            eval_topk_grid=[int(x) for x in str(args.eval_topk_grid).split(",") if x.strip().isdigit()],
            eval_threshold_grid=[float(x) for x in str(args.eval_threshold_grid).split(",") if x.strip()],
            early_metric=str(args.early_metric),
            progress=str(args.progress),
            pos_weight=pos_weight_t,
        )
    elif stage == "pseudo":
        pseudo_epochs = int(args.pseudo_epochs) if args.pseudo_epochs is not None else int(args.epochs)
        pseudo_lr = float(args.pseudo_lr) if args.pseudo_lr is not None else float(cfg.train.learning_rate)
        total_steps = (len(dl) * int(pseudo_epochs) + grad_accum - 1) // grad_accum
        optim, scheduler = _make_optim_and_sched(lr=pseudo_lr, steps=total_steps)
        _train_loop(
            model=model,
            dl=dl,
            device=device,
            optim=optim,
            scheduler=scheduler,
            scaler=scaler,
            use_fp16=use_fp16,
            grad_accum=grad_accum,
            use_confidence_weight=bool(cfg.train.use_confidence_weight),
            max_grad_norm=float(cfg.train.max_grad_norm),
            epochs=int(pseudo_epochs),
            desc_prefix="pseudo",
            val_dl=val_dl,
            patience=int(args.patience),
            min_delta=float(args.min_delta),
            eval_topk=int(args.eval_topk) if args.eval_topk is not None else None,
            eval_threshold=float(args.eval_threshold) if args.eval_threshold is not None else None,
            eval_topk_grid=[int(x) for x in str(args.eval_topk_grid).split(",") if x.strip().isdigit()],
            eval_threshold_grid=[float(x) for x in str(args.eval_threshold_grid).split(",") if x.strip()],
            early_metric=str(args.early_metric),
            progress=str(args.progress),
            pos_weight=pos_weight_t,
        )
    else:
        # both
        pre_rows = [r for r in rows if float(r.conf) >= float(args.pretrain_min_conf)][: int(args.pretrain_max_rows)]
        if len(pre_rows) == 0:
            logging.warning("Pretrain rows empty after filtering; skipping pretrain stage.")
        else:
            pre_ds = SilverDataset(pre_rows)
            pre_dl = DataLoader(pre_ds, **dl_kwargs)
            pre_steps = (len(pre_dl) * int(args.pretrain_epochs) + grad_accum - 1) // grad_accum
            optim, scheduler = _make_optim_and_sched(lr=cfg.train.learning_rate, steps=pre_steps)
            _train_loop(
                model=model,
                dl=pre_dl,
                device=device,
                optim=optim,
                scheduler=scheduler,
                scaler=scaler,
                use_fp16=use_fp16,
                grad_accum=grad_accum,
                use_confidence_weight=bool(cfg.train.use_confidence_weight),
                max_grad_norm=float(cfg.train.max_grad_norm),
                epochs=int(args.pretrain_epochs),
                desc_prefix="pretrain",
                val_dl=val_dl,
                patience=int(args.patience),
                min_delta=float(args.min_delta),
                eval_topk=int(args.eval_topk) if args.eval_topk is not None else None,
                eval_threshold=float(args.eval_threshold) if args.eval_threshold is not None else None,
                eval_topk_grid=[int(x) for x in str(args.eval_topk_grid).split(",") if x.strip().isdigit()],
                eval_threshold_grid=[float(x) for x in str(args.eval_threshold_grid).split(",") if x.strip()],
                early_metric=str(args.early_metric),
                progress=str(args.progress),
                pos_weight=pos_weight_t,
            )

        pseudo_epochs = int(args.pseudo_epochs) if args.pseudo_epochs is not None else int(args.epochs)
        pseudo_lr = float(args.pseudo_lr) if args.pseudo_lr is not None else float(cfg.train.learning_rate)
        full_ds = SilverDataset(rows)
        full_dl = DataLoader(full_ds, **dl_kwargs)
        pseudo_steps = (len(full_dl) * int(pseudo_epochs) + grad_accum - 1) // grad_accum
        optim, scheduler = _make_optim_and_sched(lr=pseudo_lr, steps=pseudo_steps)
        _train_loop(
            model=model,
            dl=full_dl,
            device=device,
            optim=optim,
            scheduler=scheduler,
            scaler=scaler,
            use_fp16=use_fp16,
            grad_accum=grad_accum,
            use_confidence_weight=bool(cfg.train.use_confidence_weight),
            max_grad_norm=float(cfg.train.max_grad_norm),
            epochs=int(pseudo_epochs),
            desc_prefix="pseudo",
            val_dl=val_dl,
            patience=int(args.patience),
            min_delta=float(args.min_delta),
            eval_topk=int(args.eval_topk) if args.eval_topk is not None else None,
            eval_threshold=float(args.eval_threshold) if args.eval_threshold is not None else None,
            eval_topk_grid=[int(x) for x in str(args.eval_topk_grid).split(",") if x.strip().isdigit()],
            eval_threshold_grid=[float(x) for x in str(args.eval_threshold_grid).split(",") if x.strip()],
            early_metric=str(args.early_metric),
            progress=str(args.progress),
            pos_weight=pos_weight_t,
        )

    ensure_dir(cfg.paths.model_dir)
    out_dir = os.path.join(cfg.paths.model_dir, "student")
    ensure_dir(out_dir)
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)

    with open(os.path.join(out_dir, "label2id.json"), "w", encoding="utf-8") as f:
        json.dump(label2id, f, ensure_ascii=False)

    logging.info("Saved model to %s", out_dir)


if __name__ == "__main__":
    main()
=======
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

    # Training 루프: Silver Label -> Self-Training -> Final Training
    
    # 1. Silver Label로 사전학습
    print("1. Pretraining with Silver Labels")
    train_dataset = GraphDataset(config.TRAIN_EMB_PATH, silver_labels, num_classes=real_num_classes)
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    
    best_loss = float('inf')
    no_improvement_count = 0
    
    for epoch in range(config.MAX_TRAINING_ITERATIONS):
        avg_loss = train_model(model, train_loader, criterion, optimizer, device)
        print(f"Pretraining Epoch {epoch+1}/{config.MAX_TRAINING_ITERATIONS}, Loss: {avg_loss:.4f}")
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            no_improvement_count = 0
        else:
            no_improvement_count += 1
            print(f"   No improvement. Count: {no_improvement_count}/{config.EARLY_STOPPING_PATIENCE}")
            
        if no_improvement_count >= config.EARLY_STOPPING_PATIENCE:
            print("Early stopping triggered for Pretraining.")
            break

    # 2. Self-Training
    print("\n2. Self-Training")
    train_emb_data = torch.load(config.TRAIN_EMB_PATH)
    all_embeddings = train_emb_data['embeddings'].to(device)
    all_pids = train_emb_data['pids']
    
    pseudo_labels = silver_labels.copy()
    best_loss = float('inf')
    no_improvement_count = 0
    
    for iteration in range(config.MAX_TRAINING_ITERATIONS):
        # Pseudo Label 생성
        model.eval()
        add_count = 0
        with torch.no_grad():
            batch_size = config.BATCH_SIZE
            for i in tqdm(range(0, all_embeddings.size(0), batch_size), desc=f"Generating Pseudo Labels (Iter {iteration+1})"):
                batch_embs = all_embeddings[i:i+batch_size]
                logits = model(batch_embs)
                probs = torch.sigmoid(logits)
                
                mask = probs > config.PSEUDO_LABEL_THRESHOLD
                for j, is_confident in enumerate(mask):
                    pid = all_pids[i+j]
                    if pid not in pseudo_labels:
                        high_conf_idxs = torch.where(is_confident)[0].cpu().tolist()
                        if high_conf_idxs:
                            pseudo_labels[pid] = high_conf_idxs
                            add_count += 1
                            
        print(f"Added {add_count} pseudo-labeled samples in iteration {iteration + 1}")
        
        # 학습
        final_dataset = GraphDataset(config.TRAIN_EMB_PATH, pseudo_labels, num_classes=real_num_classes)
        final_loader = DataLoader(final_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
        
        avg_loss = train_model(model, final_loader, criterion, optimizer, device)
        print(f"Self-Training Iteration {iteration+1}/{config.MAX_TRAINING_ITERATIONS}, Loss: {avg_loss:.4f}")
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            no_improvement_count = 0
        else:
            no_improvement_count += 1
            print(f"   No improvement. Count: {no_improvement_count}/{config.EARLY_STOPPING_PATIENCE}")
            
        if no_improvement_count >= config.EARLY_STOPPING_PATIENCE:
            print("Early stopping triggered for Self-Training.")
            break

    # 3. Final Training
    print("\n3. Final Training with Silver + Pseudo Labels")
    final_dataset = GraphDataset(config.TRAIN_EMB_PATH, pseudo_labels, num_classes=real_num_classes)
    final_loader = DataLoader(final_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    
    best_loss = float('inf')
    no_improvement_count = 0
    
    for epoch in range(config.MAX_TRAINING_ITERATIONS):
        avg_loss = train_model(model, final_loader, criterion, optimizer, device)
        print(f"Final Training Epoch {epoch+1}/{config.MAX_TRAINING_ITERATIONS}, Loss: {avg_loss:.4f}")
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            no_improvement_count = 0
        else:
            no_improvement_count += 1
            print(f"   No improvement. Count: {no_improvement_count}/{config.EARLY_STOPPING_PATIENCE}")
            
        if no_improvement_count >= config.EARLY_STOPPING_PATIENCE:
            print("Early stopping triggered for Final Training.")
            break
    
    check_dir(config.MODEL_DIR)
    model_save_path = os.path.join(config.MODEL_DIR, "final_gcn_model.pt")
    torch.save(model.state_dict(), model_save_path)
    print(f"✅ Final model saved to {model_save_path}")

if __name__ == "__main__":
    main()
>>>>>>> 8d078ba7d40ccf7a402f171edc4e82a60ef91e2d
