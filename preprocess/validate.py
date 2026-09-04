"""核对数据守恒、标签版本、跨集合泄漏以及项目的数量验收条件。"""

import argparse
import hashlib
import json
from collections import Counter
from itertools import combinations
from pathlib import Path

from config import DATA_DIR
from preprocess.clean import normalize_title


def load_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def validate_dataset(data_dir=DATA_DIR):
    root = Path(data_dir)
    cleaned_file = root / 'processed' / 'cleaned.jsonl'
    cleaned = load_jsonl(cleaned_file)
    digest = hashlib.sha256(cleaned_file.read_bytes()).hexdigest()
    subsets = {name: load_jsonl(root / f'{name}.jsonl') for name in ('train', 'val', 'test')}
    labels = json.loads((root / 'labels.json').read_text(encoding='utf-8'))
    split = json.loads((root / 'processed' / 'split_report.json').read_text(encoding='utf-8'))
    quality = json.loads((root / 'processed' / 'quality_report.json').read_text(encoding='utf-8'))
    stats = json.loads((root / 'processed' / 'statistics.json').read_text(encoding='utf-8'))
    checks = {}
    checks['raw_inputs_match_cleaning_run'] = (
        set(quality['input_sha256']) == {p.name for p in (root/'raw').glob('R*.jsonl')}
        and all((root/'raw'/name).exists() and hashlib.sha256((root/'raw'/name).read_bytes()).hexdigest() == checksum
                for name, checksum in quality['input_sha256'].items()))
    checks['artifact_versions_match'] = all(x.get('dataset_sha256') == digest for x in (labels, split, quality, stats))
    checks['count_reconciles'] = quality['raw_count'] == len(cleaned) + quality['duplicates_removed'] + quality['rejected_count']
    checks['cleaned_ids_unique'] = len({r['id'] for r in cleaned}) == len(cleaned)
    checks['cleaned_inputs_unique'] = len({r['text'] for r in cleaned}) == len(cleaned)
    label2id = labels['label2id']
    checks['contiguous_label_ids'] = sorted(label2id.values()) == list(range(len(label2id)))
    checks['label_roundtrip'] = all(labels['id2label'].get(str(v)) == k for k,v in label2id.items())
    checks['labels_cover_cleaned'] = set(label2id) == {r['label'] for r in cleaned}
    by_id = {r['id']: r for r in cleaned}
    checks['split_content_matches_cleaned'] = all(
        r['id'] in by_id and all(r.get(k) == v for k, v in by_id[r['id']].items())
        and r.get('label_id') == label2id.get(r['label'])
        for rows in subsets.values() for r in rows)
    checks['split_union_equals_cleaned'] = {r['id'] for rows in subsets.values() for r in rows} == set(by_id)
    for name, rows in subsets.items():
        checks[f'{name}_reported_count_matches'] = len(rows) == split[f'{name}_count']
        checks[f'{name}_all_classes_present'] = {r['label'] for r in rows} == set(label2id)
        counts = Counter(r['id'] for r in rows)
        checks[f'{name}_duplicates_allowed_only_in_oversampled_train'] = (
            all(c == 1 for c in counts.values()) or (name == 'train' and split['balance'] == 'oversample'))
    overlaps = {}
    keys = {'id': lambda r: r['id'], 'model_input': lambda r: r['text'],
            'doi': lambda r: r.get('doi', ''),
            'title_source': lambda r: (normalize_title(r['title']), normalize_title(r['source']))}
    for field, key in keys.items():
        values = {name: {key(r) for r in rows if key(r)} for name,rows in subsets.items()}
        for a, b in combinations(subsets, 2):
            overlaps[f'{field}:{a}-{b}'] = len(values[a] & values[b])
    checks['no_cross_split_leakage'] = not any(overlaps.values())
    counts = Counter(r['label'] for r in cleaned)
    report = {'success': all(checks.values()), 'dataset_sha256': digest,
              'checks': checks, 'cross_split_overlap': overlaps,
              'cleaned_count': len(cleaned), 'label_count': len(counts),
              'split_counts': {name: len(rows) for name, rows in subsets.items()},
              'quantity_requirements': {'at_least_100_classes': len(counts) >= 100,
                                        'at_least_3000_records': len(cleaned) >= 3000},
              'excluded_categories': quality['excluded_categories'],
              'note': '通过的是结构和数量校验，不代表弱标签正确或模型准确率已达标。'}
    (root / 'processed' / 'validation_report.json').write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    if not report['success']:
        raise ValueError(f"验收失败: {[k for k,v in checks.items() if not v]}")
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', default=str(DATA_DIR))
    args = parser.parse_args()
    print(json.dumps(validate_dataset(args.data_dir), ensure_ascii=False, indent=2))
