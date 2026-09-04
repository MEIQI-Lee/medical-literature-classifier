"""统一的模型加载与 Top-K 文本分类推理接口。"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from config import CHECKPOINT_DIR, LABELS_FILE, MODEL_CONFIG

DEFAULT_DEPLOYMENT_MODEL = "medbert_v20"
MODEL_ALIASES = {"medbert": "medbert_v20", "macbert": "macbert_v20"}

try:
    import torch as _torch
except ImportError:  # 允许在仅做 CLI/语法检查的环境导入模块
    _torch = None


@dataclass
class LoadedModel:
    model: Any
    tokenizer: Any
    labels: dict
    device: Any
    model_kind: str
    max_length: int
    checkpoint: Path


_MODEL_CACHE: dict[tuple[str, str, str], LoadedModel] = {}


def _require_torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("推理需要 torch，请先安装训练依赖") from exc
    return torch


class QwenEmbeddingClassifier((_torch.nn.Module if _torch is not None else object)):
    """Qwen embedding encoder + attention-mask mean pooling + linear head."""

    def __init__(self, encoder: Any, hidden_size: int, num_labels: int):
        if _torch is None:  # pragma: no cover
            raise RuntimeError("Qwen 分类模型需要 torch，请先安装训练依赖")
        torch = _require_torch()
        super().__init__()
        self.encoder = encoder
        self.classifier = torch.nn.Linear(hidden_size, num_labels)

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        torch = _require_torch()
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        hidden = getattr(outputs, "last_hidden_state", None)
        if hidden is None:
            hidden = outputs if torch.is_tensor(outputs) else outputs[0]
        if hidden.ndim == 2:
            pooled = hidden
        else:
            mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        logits = self.classifier(pooled)
        loss = torch.nn.functional.cross_entropy(logits, labels) if labels is not None else None
        return SimpleNamespace(loss=loss, logits=logits)


def _resolve_checkpoint(checkpoint: str | Path | None, model_type: str | None) -> Path:
    if checkpoint:
        path = Path(checkpoint).expanduser()
        if path.is_dir():
            return path
        raise FileNotFoundError(f"checkpoint 目录不存在: {path}")
    model_type = MODEL_ALIASES.get(model_type or "", model_type)
    candidates = []
    if model_type:
        candidates.extend([CHECKPOINT_DIR / model_type / "best", CHECKPOINT_DIR / model_type])
    candidates.extend([CHECKPOINT_DIR / "best", CHECKPOINT_DIR])
    for candidate in candidates:
        if (candidate / "training_config.json").exists():
            return candidate
    discovered = sorted(CHECKPOINT_DIR.glob("*/best/training_config.json"),
                        key=lambda p: p.stat().st_mtime, reverse=True)
    if discovered:
        return discovered[0].parent
    raise FileNotFoundError(f"没有找到 checkpoint，请先运行 model/train.py: {CHECKPOINT_DIR}")


def _read_labels(checkpoint: Path) -> dict:
    path = checkpoint / "labels.json"
    if not path.exists():
        path = LABELS_FILE
    if not path.exists():
        raise FileNotFoundError(f"找不到 labels.json: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if "id2label" not in payload or "code2name" not in payload:
        raise ValueError(f"labels.json 缺少 id2label/code2name: {path}")
    return payload


def load_model(model_type: str = DEFAULT_DEPLOYMENT_MODEL, checkpoint: str | Path | None = None,
               device: str | None = None) -> LoadedModel:
    """加载训练好的 checkpoint；生产推理建议显式传 checkpoint。"""
    torch = _require_torch()
    try:
        from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("推理需要 transformers，请先安装训练依赖") from exc

    selected_model = model_type or DEFAULT_DEPLOYMENT_MODEL
    cache_key = (selected_model, str(checkpoint or ""), str(device or ""))
    if cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key]
    checkpoint_path = _resolve_checkpoint(checkpoint, selected_model)
    metadata_path = checkpoint_path / "training_config.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
    kind = metadata.get("model_kind", "sequence_classification")
    trust_remote_code = bool(metadata.get("trust_remote_code", True))
    local_files_only = bool(metadata.get("local_files_only", False))
    labels = _read_labels(checkpoint_path)
    max_length = int(metadata.get("max_length", MODEL_CONFIG.get("max_length", 512)))
    selected_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    tokenizer = AutoTokenizer.from_pretrained(
        str(checkpoint_path), trust_remote_code=trust_remote_code,
        local_files_only=local_files_only,
    )
    num_labels = len(labels["label2id"])
    if kind == "embedding_classifier":
        encoder = AutoModel.from_pretrained(
            str(checkpoint_path / "encoder"), trust_remote_code=trust_remote_code,
            local_files_only=local_files_only,
        )
        hidden_size = int(getattr(encoder.config, "hidden_size",
                                  getattr(encoder.config, "embedding_size", 0)))
        if not hidden_size:
            raise ValueError("无法从 embedding encoder config 推断 hidden_size")
        model = QwenEmbeddingClassifier(encoder, hidden_size, num_labels)
        head_path = checkpoint_path / "classifier.pt"
        if not head_path.exists():
            raise FileNotFoundError(f"找不到 Qwen 分类头: {head_path}")
        model.classifier.load_state_dict(torch.load(head_path, map_location="cpu"))
    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            str(checkpoint_path), num_labels=num_labels,
            trust_remote_code=trust_remote_code, local_files_only=local_files_only,
        )
    model.to(selected_device).eval()
    loaded = LoadedModel(model, tokenizer, labels, selected_device, kind, max_length, checkpoint_path)
    _MODEL_CACHE[cache_key] = loaded
    return loaded


def classify(text: str, top_k: int = 5, model_type: str | None = None,
             checkpoint: str | Path | None = None, device: str | None = None) -> dict:
    """对一段文本返回 Top-1 和 Top-K 类别。"""
    if not isinstance(text, str) or not text.strip():
        raise ValueError("text 不能为空")
    if top_k < 1:
        raise ValueError("top_k 必须大于 0")
    selected_model = model_type or DEFAULT_DEPLOYMENT_MODEL
    loaded = load_model(selected_model, checkpoint, device)
    torch = _require_torch()
    try:
        from preprocess.clean import clean_special_chars
        text = clean_special_chars(text)
    except ImportError:
        text = " ".join(text.split())
    encoded = loaded.tokenizer(text, return_tensors="pt", truncation=True,
                               max_length=loaded.max_length)
    encoded = {key: value.to(loaded.device) for key, value in encoded.items()}
    with torch.inference_mode():
        logits = loaded.model(**encoded).logits
        probabilities = torch.softmax(logits, dim=-1)[0]
    count = min(top_k, probabilities.numel())
    scores, indices = torch.topk(probabilities, k=count)
    topk = []
    for score, index in zip(scores.tolist(), indices.tolist()):
        code = loaded.labels["id2label"].get(str(index), str(index))
        topk.append({"code": code, "name": loaded.labels["code2name"].get(code, ""),
                     "confidence": round(float(score), 6)})
    return {"top1": topk[0], "topk": topk,
            "model_type": selected_model,
            "checkpoint": str(loaded.checkpoint)}


def _main() -> None:
    parser = argparse.ArgumentParser(description="已训练文本分类模型推理")
    parser.add_argument("--checkpoint", default=None,
                        help="可选；省略时使用 model/checkpoints/medbert_v20/best")
    parser.add_argument("--text", required=True)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    print(json.dumps(classify(args.text, args.top_k, checkpoint=args.checkpoint,
                              device=args.device), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    _main()
