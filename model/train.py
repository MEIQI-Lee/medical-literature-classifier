"""预训练中文模型微调入口。

示例（AutoDL）：
  python -m model.train --model-type macbert \
    --model-path /root/autodl-tmp/models/chinese-macbert-base --epochs 3
  python -m model.train --model-type medbert \
    --model-path /root/autodl-tmp/models/medbert-base-chinese --epochs 3
  python -m model.train --model-type qwen_embedding \
    --model-path /root/autodl-tmp/models/Qwen3-Embedding-0.6B --epochs 3
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import time
from pathlib import Path
from typing import Any

from config import CHECKPOINT_DIR, DATA_DIR, LABELS_FILE, MODEL_CONFIG, MODEL_DIR, RESULTS_DIR
from model.classifier import QwenEmbeddingClassifier
from model.metrics import classification_metrics


DEFAULT_MODEL_PATHS = {
    "macbert": "/root/autodl-tmp/models/chinese-macbert-base",
    "medbert": str(MODEL_DIR / "pretrained" / "medbert-base-chinese"),
    "qwen_embedding": "/root/autodl-tmp/models/Qwen3-Embedding-0.6B",
}


def _load_jsonl(path: str | Path) -> list[dict]:
    file = Path(path)
    if not file.exists():
        raise FileNotFoundError(f"数据文件不存在: {file}")
    rows = [json.loads(line) for line in file.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"数据文件为空: {file}")
    for row in rows:
        if not isinstance(row.get("text"), str) or not row["text"].strip():
            raise ValueError(f"数据缺少 text: {file}")
        if not isinstance(row.get("label_id"), int):
            raise ValueError(f"数据缺少合法 label_id: {file}")
    return rows


def _seed_everything(seed: int) -> None:
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class TextDataset:
    def __init__(self, rows: list[dict]):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        return row["text"], row["label_id"]


def _collate(tokenizer: Any, max_length: int):
    import torch

    def collate(batch):
        texts, labels = zip(*batch)
        encoded = tokenizer(list(texts), padding=True, truncation=True,
                            max_length=max_length, return_tensors="pt")
        encoded["labels"] = torch.tensor(labels, dtype=torch.long)
        return encoded

    return collate


def _make_model(model_type: str, model_path: str, num_labels: int, labels: dict,
                trust_remote_code: bool, local_files_only: bool):
    import torch
    from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=trust_remote_code, local_files_only=local_files_only,
    )
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is None:
            raise ValueError("Tokenizer 没有 pad_token/eos_token，无法批量训练")
        tokenizer.pad_token = tokenizer.eos_token

    if model_type == "qwen_embedding":
        encoder = AutoModel.from_pretrained(
            model_path, trust_remote_code=trust_remote_code, local_files_only=local_files_only,
        )
        hidden_size = int(getattr(encoder.config, "hidden_size",
                                  getattr(encoder.config, "embedding_size", 0)))
        if not hidden_size:
            raise ValueError("无法从 Qwen config 推断 hidden_size")
        model = QwenEmbeddingClassifier(encoder, hidden_size, num_labels)
        return model, tokenizer, "embedding_classifier"

    model = AutoModelForSequenceClassification.from_pretrained(
        model_path, num_labels=num_labels, ignore_mismatched_sizes=True,
        id2label=labels["id2label"], label2id=labels["label2id"],
        trust_remote_code=trust_remote_code, local_files_only=local_files_only,
    )
    if getattr(model.config, "pad_token_id", None) is None:
        model.config.pad_token_id = tokenizer.pad_token_id
    return model, tokenizer, "sequence_classification"


def _run_epoch(model, loader, optimizer, scheduler, device, scaler, grad_accumulation: int,
               use_amp: bool, train: bool):
    import torch

    model.train(train)
    total_loss = 0.0
    labels_all: list[int] = []
    predictions: list[int] = []
    if train:
        optimizer.zero_grad(set_to_none=True)
    for step, batch in enumerate(loader):
        batch = {key: value.to(device) for key, value in batch.items()}
        with torch.cuda.amp.autocast(enabled=use_amp):
            output = model(**batch)
            loss = output.loss
            if loss is None:
                raise RuntimeError("模型没有返回 loss，请检查 labels 输入")
            scaled_loss = loss / grad_accumulation if train else loss
        if train:
            scaler.scale(scaled_loss).backward()
            if (step + 1) % grad_accumulation == 0 or step + 1 == len(loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                if scheduler is not None:
                    scheduler.step()
        total_loss += float(loss.detach().cpu())
        labels_all.extend(batch["labels"].detach().cpu().tolist())
        predictions.extend(output.logits.argmax(dim=-1).detach().cpu().tolist())
    metrics = classification_metrics(labels_all, predictions,
                                     int(model.classifier.out_features
                                         if hasattr(model, "classifier") and hasattr(model.classifier, "out_features")
                                         else model.config.num_labels))
    metrics["loss"] = total_loss / max(len(loader), 1)
    return metrics


def _save_checkpoint(model, tokenizer, labels: dict, output_dir: Path, model_kind: str,
                     model_type: str, model_path: str, max_length: int, epochs: int,
                     batch_size: int, learning_rate: float, seed: int,
                     trust_remote_code: bool, local_files_only: bool, best_metrics: dict):
    import torch

    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if model_kind == "embedding_classifier":
        model.encoder.save_pretrained(output_dir / "encoder")
        torch.save(model.classifier.state_dict(), output_dir / "classifier.pt")
    else:
        model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    (output_dir / "labels.json").write_text(json.dumps(labels, ensure_ascii=False, indent=2), encoding="utf-8")
    metadata = {
        "model_type": model_type, "model_kind": model_kind, "base_model_path": model_path,
        "max_length": max_length, "epochs": epochs, "batch_size": batch_size,
        "learning_rate": learning_rate, "seed": seed,
        "trust_remote_code": trust_remote_code, "local_files_only": local_files_only,
        "best_metrics": best_metrics,
    }
    (output_dir / "training_config.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def run_training(model_type: str = "medbert", model_path: str | None = None,
                 train_file: str | Path = DATA_DIR / "train.jsonl",
                 val_file: str | Path = DATA_DIR / "val.jsonl", epochs: int = MODEL_CONFIG["epochs"],
                 batch_size: int = 8, max_length: int = 512, learning_rate: float = 2e-5,
                 seed: int = 42, output_name: str | None = None,
                 device: str | None = None, local_files_only: bool = False,
                 trust_remote_code: bool = True, grad_accumulation: int = 1,
                 use_class_weights: bool = False) -> dict:
    """训练并保存最佳验证集 F1 checkpoint。"""
    import torch
    from torch.utils.data import DataLoader
    from transformers import get_linear_schedule_with_warmup

    if model_type not in DEFAULT_MODEL_PATHS:
        raise ValueError(f"model_type 必须是 {sorted(DEFAULT_MODEL_PATHS)}")
    if epochs < 1 or batch_size < 1 or max_length < 1 or grad_accumulation < 1:
        raise ValueError("epochs/batch_size/max_length/grad_accumulation 必须大于 0")
    labels = json.loads(Path(LABELS_FILE).read_text(encoding="utf-8"))
    num_labels = len(labels["label2id"])
    train_rows, val_rows = _load_jsonl(train_file), _load_jsonl(val_file)
    if set(row["label_id"] for row in train_rows + val_rows) - set(range(num_labels)):
        raise ValueError("数据中的 label_id 超出 labels.json")
    _seed_everything(seed)
    selected_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    base_path = model_path or DEFAULT_MODEL_PATHS[model_type]
    model, tokenizer, model_kind = _make_model(
        model_type, base_path, num_labels, labels, trust_remote_code, local_files_only,
    )
    model.to(selected_device)
    collate = _collate(tokenizer, max_length)
    train_loader = DataLoader(TextDataset(train_rows), batch_size=batch_size, shuffle=True,
                              collate_fn=collate, pin_memory=selected_device.type == "cuda")
    val_loader = DataLoader(TextDataset(val_rows), batch_size=batch_size, shuffle=False,
                            collate_fn=collate, pin_memory=selected_device.type == "cuda")
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    total_steps = (len(train_loader) + grad_accumulation - 1) // grad_accumulation * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, 0, total_steps)
    scaler = torch.cuda.amp.GradScaler(enabled=selected_device.type == "cuda")
    class_weights = None
    if use_class_weights:
        values = [labels["class_weights"].get(labels["id2label"].get(str(i)), 1.0)
                  for i in range(num_labels)]
        class_weights = torch.tensor(values, dtype=torch.float, device=selected_device)
        if model_kind == "embedding_classifier":
            # Qwen wrapper uses its own loss; class weights are intentionally ignored there.
            print("warning: qwen_embedding 当前未应用 class weights")
        else:
            original_forward = model.forward
            def weighted_forward(*args, **kwargs):
                output = original_forward(*args, **kwargs)
                if kwargs.get("labels") is not None:
                    output.loss = torch.nn.functional.cross_entropy(output.logits, kwargs["labels"], weight=class_weights)
                return output
            model.forward = weighted_forward

    run_name = output_name or f"{model_type}_v20"
    run_dir = CHECKPOINT_DIR / run_name
    best_dir = run_dir / "best"
    history = []
    best_score = -1.0
    best_metrics = None
    started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        train_metrics = _run_epoch(model, train_loader, optimizer, scheduler, selected_device,
                                   scaler, grad_accumulation, selected_device.type == "cuda", True)
        with torch.no_grad():
            val_metrics = _run_epoch(model, val_loader, optimizer, None, selected_device,
                                     scaler, grad_accumulation, False, False)
        record = {"epoch": epoch, "train": train_metrics, "val": val_metrics}
        history.append(record)
        print(f"epoch {epoch}/{epochs} train_loss={train_metrics['loss']:.4f} "
              f"val_loss={val_metrics['loss']:.4f} val_f1_macro={val_metrics['f1_macro']:.4f}")
        if val_metrics["f1_macro"] > best_score:
            best_score = val_metrics["f1_macro"]
            best_metrics = val_metrics
            _save_checkpoint(model, tokenizer, labels, best_dir, model_kind, model_type,
                             base_path, max_length, epochs, batch_size, learning_rate,
                             seed, trust_remote_code, local_files_only, best_metrics)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    history_path = RESULTS_DIR / f"{run_name}_history.json"
    history_path.write_text(json.dumps({"model_type": model_type, "checkpoint": str(best_dir),
                                        "device": str(selected_device), "elapsed_seconds": time.perf_counter() - started,
                                        "history": history}, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"model_type": model_type, "checkpoint": str(best_dir), "best_metrics": best_metrics,
            "history_file": str(history_path), "device": str(selected_device)}


def train_model(model_type: str = "medbert", train_file: str = str(DATA_DIR / "train.jsonl"),
                val_file: str = str(DATA_DIR / "val.jsonl"), epochs: int = MODEL_CONFIG["epochs"], **kwargs) -> dict:
    """兼容 Agent/旧入口的训练函数。"""
    return run_training(model_type=model_type, train_file=train_file, val_file=val_file,
                        epochs=epochs, **kwargs)


def _main() -> None:
    parser = argparse.ArgumentParser(description="中文预训练模型文本分类微调")
    parser.add_argument("--model-type", choices=sorted(DEFAULT_MODEL_PATHS), default="medbert")
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--train-file", default=str(DATA_DIR / "train.jsonl"))
    parser.add_argument("--val-file", default=str(DATA_DIR / "val.jsonl"))
    parser.add_argument("--epochs", type=int, default=MODEL_CONFIG["epochs"])
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=MODEL_CONFIG["max_length"])
    parser.add_argument("--learning-rate", type=float, default=MODEL_CONFIG["learning_rate"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-name", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--grad-accumulation", type=int, default=1)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--no-trust-remote-code", dest="trust_remote_code", action="store_false")
    parser.add_argument("--use-class-weights", action="store_true")
    args = parser.parse_args()
    result = run_training(**vars(args))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    _main()
