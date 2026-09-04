"""生成可供模型分析和 Web 展示使用的数据集统计。"""

from __future__ import annotations

import json
import hashlib
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median

from config import PROCESSED_DIR


def _percentile(sorted_values: list[int], percentile: float) -> float:
    if not sorted_values:
        return 0.0
    position = (len(sorted_values) - 1) * percentile
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return float(sorted_values[lower])
    return round(sorted_values[lower] * (upper - position) + sorted_values[upper] * (position - lower), 2)


def compute_statistics(data_file: str | Path = PROCESSED_DIR / "cleaned.jsonl",
                       output_file: str | Path = PROCESSED_DIR / "statistics.json") -> dict:
    """计算类别、缺失字段、文本长度和质量标记分布。"""
    path = Path(data_file)
    if not path.exists():
        raise FileNotFoundError(f"数据文件不存在: {path}")
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number} 不是合法 JSON: {exc}") from exc
    if not rows:
        raise ValueError(f"数据文件为空: {path}")

    category_counts = Counter(row.get("label", "") for row in rows)
    category_counts.pop("", None)
    lengths = sorted(len(str(row.get("text", ""))) for row in rows)
    missing_fields = {}
    for field in ("title", "keywords", "abstract", "source", "year", "doi"):
        count = sum(not row.get(field) for row in rows)
        missing_fields[field] = {"count": count, "rate": round(count / len(rows), 6)}
    flag_counts = Counter(flag for row in rows for flag in row.get("quality_flags", []))
    counts = list(category_counts.values())
    groups = defaultdict(list)
    for row in rows:
        groups[row['label']].append(row)
    category_quality = {}
    for label, group in sorted(groups.items()):
        category_quality[label] = {
            'name': group[0].get('category_name', ''), 'count': len(group),
            'missing_abstract': sum(not r.get('abstract') for r in group),
            'missing_keywords': sum(not r.get('keywords') for r in group),
            'title_only': sum('title_only' in r.get('quality_flags', []) for r in group),
            'possibly_truncated_abstract': sum('possibly_truncated_abstract' in r.get('quality_flags', []) for r in group),
            'mean_text_length': round(mean(len(r['text']) for r in group), 2),
        }
    boundaries = [(0, 128), (129, 256), (257, 512), (513, 1024), (1025, None)]
    histogram = {f'{lo}-{hi}' if hi else f'{lo}+': sum(n >= lo and (hi is None or n <= hi) for n in lengths)
                 for lo, hi in boundaries}
    report = {
        "data_file": str(path), "total": len(rows), "label_count": len(category_counts),
        "dataset_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "length_unit": "Unicode characters, not model tokens",
        "text_length_histogram": histogram,
        "category_quality": category_quality,
        "classes_without_abstracts": [k for k, v in category_quality.items() if v['missing_abstract'] == v['count']],
        "category_distribution": dict(sorted(category_counts.items())),
        "text_length_stats": {"min": lengths[0], "mean": round(mean(lengths), 2),
                              "median": median(lengths), "p95": _percentile(lengths, 0.95),
                              "max": lengths[-1]},
        "missing_fields": missing_fields,
        "quality_flag_distribution": dict(sorted(flag_counts.items())),
        "imbalance_ratio": round(max(counts) / min(counts), 6) if counts else 0.0,
    }
    destination = Path(output_file)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    report["output_file"] = str(destination)
    return report


if __name__ == "__main__":
    print(json.dumps(compute_statistics(), ensure_ascii=False, indent=2))
