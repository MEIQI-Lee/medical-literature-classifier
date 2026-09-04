"""在验证集或测试集上评估已保存模型。"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from config import DATA_DIR, RESULTS_DIR
from model.metrics import classification_metrics, confusion_matrix
from model.train import _collate, _load_jsonl, TextDataset
from model.classifier import load_model


def _plot_confusion(matrix: list[list[int]], labels: list[str], path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        path.with_suffix(".json").write_text(
            json.dumps({"labels": labels, "matrix": matrix}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return
    size = max(12, min(30, len(labels) // 3))
    fig, ax = plt.subplots(figsize=(size, size))
    sns.heatmap(matrix, cmap="Blues", cbar=True, xticklabels=labels,
                yticklabels=labels, ax=ax)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("Confusion matrix")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def evaluate_model(model_type: str = "medbert", test_file: str | Path = DATA_DIR / "test.jsonl",
                   checkpoint: str | Path | None = None, batch_size: int = 8,
                   device: str | None = None, output_name: str | None = None) -> dict:
    """计算 Accuracy、Macro/Micro-F1、推理耗时并生成混淆矩阵。"""
    import torch
    from torch.utils.data import DataLoader

    loaded = load_model(model_type=model_type, checkpoint=checkpoint, device=device)
    rows = _load_jsonl(test_file)
    loader = DataLoader(TextDataset(rows), batch_size=batch_size, shuffle=False,
                        collate_fn=_collate(loaded.tokenizer, loaded.max_length),
                        pin_memory=loaded.device.type == "cuda")
    labels_true: list[int] = []
    labels_pred: list[int] = []
    started = time.perf_counter()
    with torch.inference_mode():
        for batch in loader:
            batch = {key: value.to(loaded.device) for key, value in batch.items()}
            output = loaded.model(**batch)
            labels_true.extend(batch["labels"].cpu().tolist())
            labels_pred.extend(output.logits.argmax(dim=-1).cpu().tolist())
    elapsed = time.perf_counter() - started
    num_labels = len(loaded.labels["label2id"])
    metrics = classification_metrics(labels_true, labels_pred, num_labels)
    metrics.update({
        "model_type": model_type,
        "checkpoint": str(loaded.checkpoint),
        "test_file": str(test_file),
        "sample_count": len(rows),
        "inference_time_ms": round(elapsed * 1000, 3),
        "inference_time_per_sample_ms": round(elapsed * 1000 / max(len(rows), 1), 4),
        "model_size_mb": round(sum(p.stat().st_size for p in loaded.checkpoint.rglob("*") if p.is_file()) / 1024**2, 3),
    })
    label_names = [loaded.labels["id2label"].get(str(i), str(i)) for i in range(num_labels)]
    run_name = output_name or Path(loaded.checkpoint).parent.name or model_type
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    matrix_path = RESULTS_DIR / f"{run_name}_confusion_matrix.png"
    _plot_confusion(confusion_matrix(labels_true, labels_pred, num_labels), label_names, matrix_path)
    metrics["confusion_matrix_path"] = str(matrix_path)
    metrics_path = RESULTS_DIR / f"{run_name}_metrics.json"
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    metrics["metrics_path"] = str(metrics_path)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return metrics


def _main() -> None:
    parser = argparse.ArgumentParser(description="评估已训练文本分类模型")
    parser.add_argument("--model-type", default="medbert")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--test-file", default=str(DATA_DIR / "test.jsonl"))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output-name", default=None)
    args = parser.parse_args()
    evaluate_model(**vars(args))


if __name__ == "__main__":
    _main()
