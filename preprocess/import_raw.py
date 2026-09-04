"""核验并导入成员1的交付包；同名不同内容须显式允许替换，旧版本先归档。"""

import argparse
import csv
import hashlib
import json
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from config import CATEGORY_MAPPING_FILE, DATA_DIR, PROJECT_ROOT, RAW_DIR


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_rows(path):
    rows = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if line.strip():
            row = json.loads(line)
            if not isinstance(row, dict) or row.get("clc_code") != path.stem:
                raise ValueError(f"{path}:{line_no} 字段类型或文件类别不匹配")
            rows.append(row)
    return rows


def import_raw(sources, replace_existing=False):
    mapping = json.loads(CATEGORY_MAPPING_FILE.read_text(encoding="utf-8"))
    candidates = {}
    origins = {}
    source_summary = []
    repeated = 0
    for source in sources:
        folder = Path(source).resolve()
        if folder.name != "raw":
            folder = folder / "raw"
        files = sorted(folder.glob("R*.jsonl"))
        if not files:
            raise FileNotFoundError(f"没有分类 JSONL: {folder}")
        source_rows = []
        for path in files:
            if path.stem not in mapping:
                raise ValueError(f"未知类别: {path.name}")
            rows = read_rows(path)
            source_rows.extend(rows)
            digest = sha256(path)
            if path.name in candidates:
                if candidates[path.name][1] != digest:
                    raise ValueError(f"交付包之间有同名不同内容，停止导入: {path.name}")
                repeated += 1
            else:
                candidates[path.name] = (path, digest, len(rows))
            origins.setdefault(path.name, []).append(str(path))
        csv_path = folder / "cnki_papers.csv"
        verified = None
        if csv_path.exists():
            with csv_path.open(encoding="utf-8-sig", newline="") as handle:
                csv_rows = list(csv.DictReader(handle))
            canonical = lambda row: json.dumps(row, ensure_ascii=False, sort_keys=True)
            verified = Counter(map(canonical, source_rows)) == Counter(map(canonical, csv_rows))
            if not verified:
                raise ValueError(f"CSV与该目录JSONL逐字段不一致: {csv_path}")
            digest = sha256(csv_path)
            if csv_path.name in candidates and candidates[csv_path.name][1] != digest:
                raise ValueError("交付包包含不同版本合并CSV，需先明确使用版本")
            candidates[csv_path.name] = (csv_path, digest, len(csv_rows))
            origins.setdefault(csv_path.name, []).append(str(csv_path))
        source_summary.append({"folder": str(folder), "category_files": len(files),
                               "rows": len(source_rows), "csv_matches_jsonl": verified})

    replaced = [name for name, (_, digest, _) in candidates.items()
                if (RAW_DIR / name).exists() and sha256(RAW_DIR / name) != digest]
    if replaced and not replace_existing:
        raise ValueError(f"需允许替换（自动备份）: {replaced}")

    # 所有源数据校验完成后才创建备份和修改项目数据；外部交付包始终只读。
    snapshot = DATA_DIR / "archive" / datetime.now(timezone.utc).strftime("import_%Y%m%dT%H%M%S%fZ")
    snapshot.mkdir(parents=True, exist_ok=False)
    backups = []
    for directory in (RAW_DIR, DATA_DIR / "processed"):
        if directory.exists():
            backups.extend(path for path in directory.rglob("*") if path.is_file())
    backups.extend(path for path in DATA_DIR.iterdir() if path.is_file())
    for path in backups:
        destination = snapshot / path.relative_to(DATA_DIR)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
        if sha256(path) != sha256(destination):
            raise IOError(f"备份校验失败: {path}")
    for name in ("MEMBER2_HANDOFF.md", "preprocess/README.md"):
        path = PROJECT_ROOT / name
        if path.exists():
            destination = snapshot / "docs" / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination)

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    manifest = []
    for name, (path, digest, count) in sorted(candidates.items()):
        target = RAW_DIR / name
        unchanged = target.exists() and sha256(target) == digest
        if not unchanged:
            shutil.copy2(path, target)
        if sha256(target) != digest:
            raise IOError(f"复制后校验失败: {target}")
        manifest.append({"file": name, "sources": origins[name], "sha256": digest,
                         "rows": count, "action": "unchanged" if unchanged else
                         ("replaced_with_backup" if name in replaced else "added")})
    report = {"sources": source_summary, "identical_category_files_skipped": repeated,
              "backup_directory": str(snapshot), "replaced_files": replaced,
              "imported_category_count": sum(n.endswith('.jsonl') for n in candidates),
              "imported_jsonl_rows": sum(v[2] for n, v in candidates.items() if n.endswith('.jsonl')),
              "files": manifest}
    output = DATA_DIR / "import_manifest.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (snapshot / "import_manifest.json").write_text(output.read_text(encoding="utf-8"), encoding="utf-8")
    return {k: v for k, v in report.items() if k != "files"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", nargs="+", required=True)
    parser.add_argument("--replace-existing", action="store_true")
    args = parser.parse_args()
    print(json.dumps(import_raw(args.source, args.replace_existing), ensure_ascii=False, indent=2))
