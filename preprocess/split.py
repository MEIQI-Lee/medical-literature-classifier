"""按类别分层划分训练集、验证集和测试集。"""

from __future__ import annotations

import json
import hashlib
import random
from collections import defaultdict
from pathlib import Path

from config import CATEGORY_MAPPING_FILE, PROCESSED_DIR, TRAIN_FILE


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"清洗后数据不存在: {path}")
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number} 不是合法 JSON: {exc}") from exc
            if not isinstance(value, dict) or any(not isinstance(value.get(k), str) or not value[k].strip()
                                                 for k in ('id', 'text', 'label')):
                raise ValueError(f"{path}:{line_number} 缺少合法的 id、text 或 label")
            rows.append(value)
    if not rows:
        raise ValueError(f"清洗后数据为空: {path}")
    return rows


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _split_counts(size: int, test_size: float, val_size: float) -> tuple[int, int, int]:
    if size == 1:
        return 1, 0, 0
    test_count = max(1, int(round(size * test_size))) if test_size else 0
    val_count = max(1, int(round(size * val_size))) if val_size and size >= 3 else 0
    while test_count + val_count >= size:
        if test_count >= val_count and test_count > 0:
            test_count -= 1
        elif val_count > 0:
            val_count -= 1
    return size - test_count - val_count, val_count, test_count


def _class_weights(rows: list[dict]) -> dict[str, float]:
    counts: dict[str, int] = defaultdict(int)
    for row in rows:
        counts[row["label"]] += 1
    total = len(rows)
    classes = len(counts)
    return {label: round(total / (classes * count), 8) for label, count in sorted(counts.items())}


def _oversample(rows: list[dict], random_state: int) -> list[dict]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[row["label"]].append(row)
    target = max(map(len, groups.values()))
    result = list(rows)
    for label, group in sorted(groups.items()):
        rng = random.Random(f"oversample:{random_state}:{label}")
        for index in range(target - len(group)):
            duplicate = dict(rng.choice(group))
            duplicate["oversampled"] = True
            duplicate["sample_instance"] = index + 1
            result.append(duplicate)
    return result


def split_dataset(cleaned_file: str | Path = PROCESSED_DIR / "cleaned.jsonl",
                  test_size: float = 0.2,
                  val_size: float = 0.1,
                  balance: str = "none",
                  random_state: int = 42,
                  output_dir: str | Path | None = None) -> dict:
    """执行确定性的按类分层划分，平衡操作仅作用于训练集。"""
    if not 0 <= test_size < 1 or not 0 <= val_size < 1 or test_size + val_size >= 1:
        raise ValueError("test_size 与 val_size 必须在 [0,1) 且两者之和小于1")
    if balance not in {"none", "oversample", "class_weight"}:
        raise ValueError("balance 只能为 none、oversample 或 class_weight")

    rows = _read_jsonl(Path(cleaned_file))
    for key in ('id', 'text'):
        if len({r[key] for r in rows}) != len(rows):
            raise ValueError(f"清洗数据中 {key} 重复，划分前必须先去重/隔离冲突")
    rows.sort(key=lambda r: r['id'])
    labels = sorted({row["label"] for row in rows})
    label2id = {label: index for index, label in enumerate(labels)}
    mapping = json.loads(CATEGORY_MAPPING_FILE.read_text(encoding="utf-8"))
    if set(labels) - set(mapping):
        raise ValueError(f"未知训练标签: {sorted(set(labels) - set(mapping))}")
    destination = Path(output_dir) if output_dir else TRAIN_FILE.parent
    destination.mkdir(parents=True, exist_ok=True)
    train_file, val_file, test_file = (destination / f'{name}.jsonl' for name in ('train', 'val', 'test'))
    labels_file = destination / 'labels.json'
    fingerprint = hashlib.sha256(Path(cleaned_file).read_bytes()).hexdigest()
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        item = dict(row)
        item["label_id"] = label2id[item["label"]]
        groups[item["label"]].append(item)

    train: list[dict] = []
    val: list[dict] = []
    test: list[dict] = []
    warnings = []
    distribution = {}
    for label in labels:
        group = groups[label]
        random.Random(f"split:{random_state}:{label}").shuffle(group)
        train_count, val_count, test_count = _split_counts(len(group), test_size, val_size)
        test.extend(group[:test_count])
        val.extend(group[test_count:test_count + val_count])
        train.extend(group[test_count + val_count:])
        distribution[label] = {"total": len(group), "train": train_count,
                               "val": val_count, "test": test_count}
        if val_count == 0 or test_count == 0:
            warnings.append(f"类别 {label} 样本仅 {len(group)} 条，无法同时覆盖验证集和测试集")

    class_weights = _class_weights(train)
    original_train_count = len(train)
    if balance == "oversample":
        train = _oversample(train, random_state)
    for name, values in (("train", train), ("val", val), ("test", test)):
        random.Random(f"final:{random_state}:{name}").shuffle(values)

    _write_jsonl(train_file, train)
    _write_jsonl(val_file, val)
    _write_jsonl(test_file, test)
    labels_payload = {
        "label2id": label2id,
        "id2label": {str(index): label for label, index in label2id.items()},
        "code2name": {label: mapping.get(label, "") for label in labels},
        "class_weights": class_weights,
        "dataset_sha256": fingerprint,
        "class_weight_basis": "training set before resampling; applied by training code only when requested",
        "balance": balance,
    }
    labels_file.write_text(json.dumps(labels_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    report = {
        "source_file": str(cleaned_file), "random_state": random_state,
        "dataset_sha256": fingerprint,
        "test_size": test_size, "val_size": val_size, "balance": balance,
        "label_count": len(labels), "train_count_before_balance": original_train_count,
        "train_count": len(train), "val_count": len(val), "test_count": len(test),
        "distribution": distribution, "warnings": warnings,
        "train_file": str(train_file), "val_file": str(val_file),
        "test_file": str(test_file), "labels_file": str(labels_file),
    }
    report_file = destination / 'processed' / "split_report.json"
    report_file.parent.mkdir(parents=True, exist_ok=True)
    report_file.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    report["report_file"] = str(report_file)
    return report


if __name__ == "__main__":
    print(json.dumps(split_dataset(), ensure_ascii=False, indent=2))
