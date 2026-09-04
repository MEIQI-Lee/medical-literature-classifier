"""Load member 3's deployment bundle and expose the member 4 model protocol."""

import hashlib
import json
from pathlib import Path

from agent.catalog import CategoryCatalog
from agent.errors import AgentError
from agent.integrations.team_preprocessor import PREPROCESS_VERSION
from agent.schemas import (
    Category,
    CategoryMapping,
    LabelEntry,
    LabelMapping,
    ModelInfo,
    ModelOutput,
    ModelScore,
)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CHECKPOINT = ROOT / "model/checkpoints/medbert_v20/best"


def _digest(value) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()
    ).hexdigest()


def inspect_bundle(checkpoint_path: str | Path = DEFAULT_CHECKPOINT) -> dict:
    """Read deployment metadata without importing torch or starting training."""
    checkpoint = Path(checkpoint_path).expanduser().resolve()
    required = (
        "model.safetensors",
        "config.json",
        "labels.json",
        "training_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.txt",
        "special_tokens_map.json",
    )
    missing = [
        name
        for name in required
        if not (checkpoint / name).is_file() or (checkpoint / name).stat().st_size == 0
    ]
    if missing:
        raise AgentError("MODEL_NOT_READY", "部署模型缺少必要文件：" + "、".join(missing))
    try:
        labels = json.loads((checkpoint / "labels.json").read_text(encoding="utf-8"))
        config = json.loads((checkpoint / "config.json").read_text(encoding="utf-8"))
        training = json.loads((checkpoint / "training_config.json").read_text(encoding="utf-8"))
        label2id, id2label, names = labels["label2id"], labels["id2label"], labels["code2name"]
        count = len(label2id)
        if count < 2 or any(type(value) is not int for value in label2id.values()):
            raise ValueError("invalid label IDs")
        if set(label2id.values()) != set(range(count)):
            raise ValueError("non-contiguous label IDs")
        if id2label != {str(value): code for code, value in label2id.items()}:
            raise ValueError("inconsistent mappings")
        if any(not isinstance(names.get(code), str) or not names[code] for code in label2id):
            raise ValueError("missing category name")
        if config["id2label"] != id2label or config["label2id"] != label2id:
            raise ValueError("model config and labels disagree")
        if training.get("model_kind") != "sequence_classification":
            raise ValueError("expected sequence classification checkpoint")
        if training.get("local_files_only") is not True:
            raise ValueError("deployment must use local assets")
        max_length = int(training["max_length"])
        if not 2 <= max_length <= config["max_position_embeddings"]:
            raise ValueError("invalid max_length")
    except (ValueError, KeyError, TypeError, OSError) as exc:
        raise AgentError(
            "LABEL_MAPPING_INVALID", "模型标签或部署配置不一致，请检查 checkpoint。"
        ) from exc
    weight_hash = hashlib.sha256()
    with (checkpoint / "model.safetensors").open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            weight_hash.update(chunk)
    # Dataset fingerprints are informational, independent from label identity.
    current_fingerprint = None
    current_labels = ROOT / "data/labels.json"
    if current_labels.is_file():
        try:
            current_fingerprint = json.loads(current_labels.read_text(encoding="utf-8")).get(
                "dataset_sha256"
            )
        except (ValueError, OSError):
            pass
    mapping = {
        "label2id": label2id,
        "id2label": id2label,
        "code2name": {code: names[code] for code in label2id},
    }
    return {
        "checkpoint": str(checkpoint),
        "model_type": checkpoint.parent.name,
        "model_version": checkpoint.parent.name + "-" + weight_hash.hexdigest()[:12],
        "weights_sha256": weight_hash.hexdigest(),
        "label_version": "clc-" + _digest(mapping)[:16],
        "preprocess_version": PREPROCESS_VERSION,
        "labels": labels,
        "label_count": count,
        "max_length": max_length,
        "trained_dataset_sha256": labels.get("dataset_sha256"),
        "current_dataset_sha256": current_fingerprint,
        "dataset_fingerprint_match": labels.get("dataset_sha256") == current_fingerprint,
        "checkpoint_transformers_version": config.get("transformers_version"),
    }


def catalog_from_bundle(bundle: dict) -> CategoryCatalog:
    mapping = bundle["labels"]
    category_version = "clc-names-" + _digest(mapping["code2name"])[:16]
    return CategoryCatalog(
        LabelMapping(
            version=bundle["label_version"],
            category_version=category_version,
            labels=[
                LabelEntry(label_id=index, category_code=code)
                for code, index in mapping["label2id"].items()
            ],
        ),
        CategoryMapping(
            version=category_version,
            categories=[
                Category(code=code, name=mapping["code2name"][code]) for code in mapping["label2id"]
            ],
        ),
    )


class MedBertBackend:
    def __init__(self, checkpoint_path=DEFAULT_CHECKPOINT, *, device=None):
        self.bundle = inspect_bundle(checkpoint_path)
        try:
            from model.classifier import load_model

            self.loaded = load_model(
                model_type=self.bundle["model_type"],
                checkpoint=self.bundle["checkpoint"],
                device=device,
            )
        except Exception as exc:
            raise AgentError(
                "MODEL_NOT_READY", "无法加载本地 MedBERT，请检查模型依赖、权重和设备配置。"
            ) from exc
        self._info = ModelInfo(
            model_version=self.bundle["model_version"],
            label_version=self.bundle["label_version"],
            preprocess_version=PREPROCESS_VERSION,
            label_ids=list(range(self.bundle["label_count"])),
            score_type="probability",
            calibrated=False,
        )

    def describe(self) -> ModelInfo:
        return self._info.model_copy(deep=True)

    def predict(self, text: str, top_k: int = 5) -> ModelOutput:
        import torch

        if not isinstance(text, str) or not text.strip():
            raise AgentError("EMPTY_TEXT", "模型输入不能为空。")
        if type(top_k) is not int or top_k < 1:
            raise AgentError("INVALID_INPUT", "top_k 必须是正整数。")
        # Match training: tokenize the prepared string verbatim, with no second NFKC pass.
        all_ids = self.loaded.tokenizer(
            text,
            add_special_tokens=True,
            truncation=False,
            verbose=False,
        )["input_ids"]
        encoded = self.loaded.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.loaded.max_length,
        )
        encoded = {key: value.to(self.loaded.device) for key, value in encoded.items()}
        with torch.inference_mode():
            logits = self.loaded.model(**encoded).logits
            if tuple(logits.shape) != (1, self.bundle["label_count"]):
                raise AgentError("INVALID_MODEL_OUTPUT", "模型输出维度与标签数量不一致。")
            probabilities = torch.softmax(logits.float(), dim=-1)[0].cpu().tolist()
        # Sort the full vector for deterministic ties and retain original precision.
        ranked = sorted(enumerate(probabilities), key=lambda item: (-item[1], item[0]))[:top_k]
        return ModelOutput(
            model_info=self.describe(),
            predictions=[ModelScore(label_id=index, score=score) for index, score in ranked],
            input_tokens=len(all_ids),
            truncated=len(all_ids) > self.loaded.max_length,
        )


def create_classifier(checkpoint_path=DEFAULT_CHECKPOINT, *, device=None):
    return MedBertBackend(checkpoint_path, device=device)
