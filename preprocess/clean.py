"""原始知网文献清洗、去重与模型文本构造。"""

from __future__ import annotations

import hashlib
import html
import json
import re
import unicodedata
from collections import Counter, defaultdict
from itertools import combinations
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, List

from config import CATEGORY_MAPPING_FILE, PROCESSED_DIR, RAW_DIR


RAW_FIELDS = (
    "title", "authors", "orgs", "source", "year", "volume", "issue",
    "pages", "keywords", "abstract", "doi", "cited_count",
    "download_count", "fund", "classification", "clc_code", "url", "uid",
)


def clean_html(text: Any) -> str:
    """仅删除明确的HTML标签，不能把 P<0.05 ... P>0.05 当成标签。"""
    value = "" if text is None else str(text)
    value = html.unescape(html.unescape(value))
    value = re.sub(r"(?is)<(script|style)\b[^>]*>.*?</\1\s*>", " ", value)
    value = re.sub(r"(?s)<!--.*?-->", " ", value)
    tags = "a|b|br|div|em|font|h[1-6]|hr|i|img|li|ol|p|span|strong|sub|sup|table|tbody|td|th|tr|u|ul"
    return re.sub(rf"(?is)</?(?:{tags})(?=[\s/>])[^>]*>", " ", value)


def clean_special_chars(text: Any) -> str:
    """统一 Unicode、不可见字符和连续空白，保留有语义的中英文标点。"""
    value = unicodedata.normalize("NFKC", clean_html(text))
    value = value.translate({
        ord("\u200b"): None, ord("\u200c"): None, ord("\u200d"): None,
        ord("\ufeff"): None, ord("\u00ad"): None,
    })
    value = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def normalize_title(text: Any) -> str:
    """生成仅用于判重的标题形式，不修改最终展示标题。"""
    return "".join(char for char in clean_special_chars(text).casefold() if char.isalnum())


def normalize_doi(value: Any) -> str:
    doi = clean_special_chars(value).casefold()
    return re.sub(r"^(?:https?://(?:dx\.)?doi\.org/|doi\s*:\s*)", "", doi).strip()


def normalize_year(value: Any) -> int | None:
    """从年份、日期或日期时间中提取四位年份。"""
    text = clean_special_chars(value)
    if re.fullmatch(r'(?:19|20)\d{4}(?:\d{2})?', text):
        try:
            return datetime.strptime(text, '%Y%m' if len(text) == 6 else '%Y%m%d').year
        except ValueError:
            return None
    match = re.search(r"(?<!\d)((?:19|20)\d{2})(?!\d)", text)
    return int(match.group(1)) if match else None


def parse_keywords(value: Any) -> List[str]:
    """把关键词字符串或列表转换为有序、去重后的字符串列表。"""
    if isinstance(value, (list, tuple, set)):
        candidates: Iterable[Any] = value
    else:
        candidates = re.split(r"\s*[;；|｜]\s*|\s{2,}", clean_special_chars(value))
    result: List[str] = []
    seen = set()
    for item in candidates:
        keyword = clean_special_chars(item).strip(" ,，;；。")
        key = keyword.casefold()
        if keyword and key not in seen:
            result.append(keyword)
            seen.add(key)
    return result


def map_clc_to_label(clc_code: Any, mapping: dict) -> str:
    """把分类号映射到映射表中最具体的已有类别，无法映射时返回空串。"""
    code = clean_special_chars(clc_code).upper().replace(" ", "")
    if not re.fullmatch(r"R[0-9]+(?:[.\-][0-9]+)*|R-[0-9]+", code):
        return ""
    if code in mapping:
        return code
    for end in range(len(code) - 1, 0, -1):
        if code[:end] in mapping:
            return code[:end]
    return ""


def build_text_input(title: str, keywords: List[str] | str, abstract: str) -> str:
    """使用带字段提示的“标题 + 关键词 + 摘要”构造模型输入。"""
    parts = []
    title = clean_special_chars(title)
    abstract = clean_special_chars(abstract)
    keyword_list = parse_keywords(keywords)
    if title:
        parts.append(f"标题：{title}")
    if keyword_list:
        parts.append(f"关键词：{'、'.join(keyword_list)}")
    if abstract:
        parts.append(f"摘要：{abstract}")
    return "。".join(parts)


def _stable_id(record: dict) -> str:
    identity = clean_special_chars(record.get("doi")).casefold()
    if not identity:
        identity = f"{normalize_title(record.get('title'))}|{normalize_title(record.get('source'))}"
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16]


def _deduplicate_groups(records: list[dict]) -> list[list[dict]]:
    """联合论文身份判重与完全相同模型输入判重（不猜测冲突标签）。"""
    parents = list(range(len(records)))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parents[right_root] = left_root

    seen: dict[str, int] = {}
    for index, record in enumerate(records):
        title_source = (
            f"title_source:{normalize_title(record.get('title'))}|"
            f"{normalize_title(record.get('source'))}"
        )
        model_text = build_text_input(record['title'], record['keywords'], record['abstract'])
        keys = [title_source, 'input:' + model_text]
        doi = normalize_doi(record.get("doi"))
        if doi:
            keys.append(f"doi:{doi}")
        for key in keys:
            if key in seen:
                union(index, seen[key])
            else:
                seen[key] = index

    grouped: dict[int, list[dict]] = defaultdict(list)
    for index, record in enumerate(records):
        grouped[find(index)].append(record)
    return list(grouped.values())


def _completeness(record: dict) -> tuple[int, int]:
    fields = ("abstract", "keywords", "doi", "authors", "orgs", "fund", "url")
    present = sum(bool(record.get(field)) for field in fields)
    size = len(str(record.get("abstract", ""))) + sum(map(len, record.get("keywords", [])))
    return present, size


def _clean_record(raw: dict, mapping: dict, source_file: str, line_number: int) -> dict:
    record = {field: clean_special_chars(raw.get(field, "")) for field in RAW_FIELDS}
    record["keywords"] = parse_keywords(raw.get("keywords", ""))
    record["doi"] = normalize_doi(raw.get("doi", ""))
    record["year_raw"] = record["year"]
    record["year"] = normalize_year(record["year"])
    record["label"] = map_clc_to_label(record.pop("clc_code"), mapping)
    record["clc_code_raw"] = raw.get("clc_code", "")
    record["label_source"] = "search_clc_code"
    record["category_name"] = mapping.get(record["label"], "")
    record["original_uid"] = record.pop("uid")
    record["source_file"] = source_file
    record["source_line"] = line_number
    return record


def _merge_duplicate_group(records: list[dict]) -> dict:
    """保留最完整记录，并用同组记录补齐其空字段。"""
    ordered = sorted(records, key=_completeness, reverse=True)
    merged = dict(ordered[0])
    for candidate in ordered[1:]:
        for field, value in candidate.items():
            if field == "keywords":
                merged[field] = parse_keywords([*merged.get(field, []), *value])
            elif not merged.get(field) and value:
                merged[field] = value
    merged["duplicate_count"] = len(records) - 1
    merged["provenance"] = [
        {key: row[key] for key in ('source_file', 'source_line', 'original_uid', 'clc_code_raw')}
        for row in records
    ]
    return merged


def _quality_flags(record: dict) -> list[str]:
    flags = []
    if not record.get("abstract"):
        flags.append("missing_abstract")
    if not record.get("keywords"):
        flags.append("missing_keywords")
    abstract = str(record.get("abstract", "")).rstrip()
    if abstract.endswith("...") or abstract.endswith("…"):
        flags.append("possibly_truncated_abstract")
    if not record.get("year"):
        flags.append("invalid_or_missing_year")
    if not record.get("doi"):
        flags.append("missing_doi")
    if not record.get("source"):
        flags.append("missing_source")
    if not record.get("abstract") and not record.get("keywords"):
        flags.append("title_only")
        if len(normalize_title(record.get("title"))) < 8:
            flags.append("short_title_only")
    if not record.get("classification"):
        flags.append("unverified_search_label")
    return flags


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def clean_raw(raw_dir: str | Path = RAW_DIR,
              output_dir: str | Path = PROCESSED_DIR) -> dict:
    """清洗 ``R*.jsonl``，输出 ``cleaned.jsonl`` 和质量报告。"""
    raw_path = Path(raw_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    mapping = json.loads(CATEGORY_MAPPING_FILE.read_text(encoding="utf-8"))
    input_files = sorted(raw_path.glob("R*.jsonl"))
    if not input_files:
        raise FileNotFoundError(f"未在 {raw_path} 找到 R*.jsonl 原始数据")

    parsed_count = 0
    raw_distribution = Counter()
    raw_missing = Counter()
    valid_records: list[dict] = []
    rejected: list[dict] = []
    malformed_json = 0
    for source in input_files:
        with source.open("r", encoding="utf-8-sig") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                parsed_count += 1
                raw_distribution[source.stem] += 1
                try:
                    raw = json.loads(line)
                except json.JSONDecodeError as exc:
                    malformed_json += 1
                    rejected.append({"source_file": source.name, "source_line": line_number,
                                     "reason": "malformed_json", "detail": str(exc)})
                    continue
                if not isinstance(raw, dict):
                    rejected.append({"source_file": source.name, "source_line": line_number,
                                     "reason": "record_is_not_object"})
                    continue
                raw_missing.update(field for field in RAW_FIELDS if not raw.get(field))
                record = _clean_record(raw, mapping, source.name, line_number)
                if not normalize_title(record["title"]):
                    rejected.append({"source_file": source.name, "source_line": line_number,
                                     "reason": "missing_title"})
                elif not record["label"]:
                    rejected.append({"source_file": source.name, "source_line": line_number,
                                     "reason": "unknown_label", "value": raw.get("clc_code", "")})
                else:
                    valid_records.append(record)

    cleaned: list[dict] = []
    label_conflicts = 0
    duplicates_removed = 0
    conflicts = []
    duplicate_audit = []
    overlap = Counter()
    for records in _deduplicate_groups(valid_records):
        labels = sorted({record["label"] for record in records})
        if len(labels) > 1:
            label_conflicts += 1
            group_id = f"conflict-{label_conflicts:04d}"
            for row in records:
                rejected.append({"reason": "conflicting_labels", "group_id": group_id,
                                 "labels": labels, "record": row,
                                 "source_file": row['source_file'], "source_line": row['source_line']})
            conflicts.append({"group_id": group_id, "title": records[0]["title"],
                              "labels": labels, "record_count": len(records),
                              "sources": [{"file": r['source_file'], "line": r['source_line']}
                                          for r in records]})
            overlap.update('|'.join(pair) for pair in combinations(labels, 2))
            continue
        duplicates_removed += len(records) - 1
        record = _merge_duplicate_group(records)
        record["id"] = _stable_id(record)
        record["text"] = build_text_input(record["title"], record["keywords"], record["abstract"])
        record["quality_flags"] = _quality_flags(record)
        cleaned.append(record)
        if len(records) > 1:
            duplicate_audit.append({"id": record['id'], "label": record['label'],
                                    "title": record['title'], "removed": len(records) - 1,
                                    "provenance": record['provenance']})

    cleaned.sort(key=lambda item: (item["label"], item["id"]))
    cleaned_file = out_path / "cleaned.jsonl"
    rejected_file = out_path / "rejected.jsonl"
    _write_jsonl(cleaned_file, cleaned)
    _write_jsonl(rejected_file, rejected)
    _write_jsonl(out_path / "label_conflicts.jsonl", conflicts)
    _write_jsonl(out_path / "duplicate_groups.jsonl", duplicate_audit)

    clean_distribution = Counter(row['label'] for row in cleaned)
    fingerprint = hashlib.sha256(cleaned_file.read_bytes()).hexdigest()

    report = {
        "input_files": [path.name for path in input_files],
        "input_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in input_files},
        "raw_count": parsed_count,
        "schema_version": "2.0",
        "dataset_sha256": fingerprint,
        "raw_label_count": len(raw_distribution),
        "cleaned_label_count": len(clean_distribution),
        "raw_category_distribution": dict(sorted(raw_distribution.items())),
        "cleaned_category_distribution": dict(sorted(clean_distribution.items())),
        "excluded_categories": sorted(set(raw_distribution) - set(clean_distribution)),
        "raw_missing_fields": dict(sorted(raw_missing.items())),
        "normalized_year_count": sum(r['year'] is not None and str(r['year']) != r['year_raw']
                                     for r in valid_records),
        "valid_before_dedup": len(valid_records),
        "cleaned_count": len(cleaned),
        "duplicates_removed": duplicates_removed,
        "rejected_count": len(rejected),
        "malformed_json_count": malformed_json,
        "label_conflict_groups": label_conflicts,
        "label_conflict_records": sum(c['record_count'] for c in conflicts),
        "conflict_category_pairs": dict(sorted(overlap.items())),
        "rejection_reasons": dict(Counter(row['reason'] for row in rejected)),
        "count_reconciles": parsed_count == len(cleaned) + duplicates_removed + len(rejected),
        "label_policy": "search_clc_code; quarantine all conflicting labels; never infer ground truth",
        "possibly_truncated_abstracts": sum("possibly_truncated_abstract" in row["quality_flags"] for row in cleaned),
        "missing_abstracts": sum("missing_abstract" in row["quality_flags"] for row in cleaned),
        "output_file": str(cleaned_file),
        "rejected_file": str(rejected_file),
    }
    report_file = out_path / "quality_report.json"
    report_file.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    report["report_file"] = str(report_file)
    return report


if __name__ == "__main__":
    print(json.dumps(clean_raw(), ensure_ascii=False, indent=2))
