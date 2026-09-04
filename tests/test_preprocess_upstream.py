"""成员2预处理模块的标准库单元测试。"""

import json
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

from preprocess.clean import (build_text_input, clean_raw, clean_special_chars,
                              map_clc_to_label, normalize_title, normalize_year, parse_keywords)
from preprocess.split import split_dataset
from preprocess.statistics import compute_statistics
from preprocess.validate import validate_dataset


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows), encoding='utf-8')


def read_rows(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


class CleanFunctionsTest(unittest.TestCase):
    def test_year_and_keywords(self):
        self.assertEqual(normalize_year("2026-08-20 10:00"), 2026)
        self.assertIsNone(normalize_year("未知"))
        self.assertEqual(parse_keywords("结核病 ; 筛查；结核病"), ["结核病", "筛查"])

    def test_build_text(self):
        value = build_text_input("标题", ["甲", "乙"], "摘要")
        self.assertEqual(value, "标题：标题。关键词：甲、乙。摘要：摘要")

    def test_comparison_symbols_survive_html_cleaning(self):
        text = '组间P<0.05，而另一组P>0.05，IL-6与β受体有关'
        self.assertEqual(clean_special_chars(text), text.replace('，', ','))
        self.assertEqual(clean_special_chars('P&lt;0.05，P&gt;0.05'), 'P<0.05,P>0.05')

    def test_html_entities_scripts_and_invisible_characters(self):
        self.assertEqual(clean_special_chars('&lt;b&gt;标题&lt;/b&gt;<script>alert(1)</script>\u200b'), '标题')
        self.assertIn('β', normalize_title('β受体'))

    def test_compact_years(self):
        self.assertEqual(normalize_year('202304'), 2023)
        self.assertEqual(normalize_year('20240229'), 2024)
        self.assertIsNone(normalize_year('202313'))
        self.assertIsNone(normalize_year('20230230'))

    def test_label_parser(self):
        mapping = {'R54': '心血管', 'R-33': '实验', 'R33': '生理'}
        self.assertEqual(map_clc_to_label('r541.1', mapping), 'R54')
        self.assertEqual(map_clc_to_label('R-33', mapping), 'R-33')
        self.assertNotEqual(map_clc_to_label('R-33', mapping), map_clc_to_label('R33', mapping))
        self.assertEqual(map_clc_to_label('R54;R51', mapping), '')
        self.assertEqual(map_clc_to_label('R54garbage', mapping), '')

    def test_conflicts_are_quarantined_with_full_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            a = {'title': '同一论文', 'source': '期刊', 'clc_code': 'R51'}
            b = dict(a, clc_code='R52')
            write_rows(root/'raw/R51.jsonl', [a])
            write_rows(root/'raw/R52.jsonl', [b])
            report = clean_raw(root/'raw', root/'processed')
            self.assertEqual(report['cleaned_count'], 0)
            self.assertEqual(report['duplicates_removed'], 0)
            self.assertEqual(report['rejected_count'], 2)
            self.assertTrue(report['count_reconciles'])
            rejected = read_rows(root/'processed/rejected.jsonl')
            self.assertEqual({r['record']['label'] for r in rejected}, {'R51', 'R52'})
            self.assertTrue(all(r['source_line'] == 1 for r in rejected))

    def test_identical_inputs_from_different_sources_merge(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            a = {'title': '重复转载论文标题', 'source': '日报', 'clc_code': 'R51'}
            write_rows(root/'raw/R51.jsonl', [a, dict(a, source='晚报')])
            report = clean_raw(root/'raw', root/'processed')
            self.assertEqual(report['cleaned_count'], 1)
            self.assertEqual(report['duplicates_removed'], 1)

    def test_malformed_missing_and_unknown_rows_accounted_for(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root/'raw/R51.jsonl'
            write_rows(path, [{}, [], {'title': '未知标签', 'clc_code': 'INVALID'}])
            with path.open('a', encoding='utf-8') as stream:
                stream.write('{broken\n')
            report = clean_raw(root/'raw', root/'processed')
            self.assertEqual(report['rejected_count'], 4)
            self.assertTrue(report['count_reconciles'])


class SplitIntegrationTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        for label, count in [('R31', 29), ('R51', 50), ('R52', 50)]:
            write_rows(self.root/f'raw/{label}.jsonl', [
                {'clc_code': label, 'title': f'{label}论文{i}', 'source': '期刊',
                 'abstract': f'{label}摘要{i}', 'year': '2026'} for i in range(count)])
        self.cleaned = self.root/'processed/cleaned.jsonl'
        clean_raw(self.root/'raw', self.root/'processed')
        compute_statistics(self.cleaned, self.root/'processed/statistics.json')

    def run_split(self, **kwargs):
        return split_dataset(self.cleaned, output_dir=self.root, **kwargs)

    def test_r31_counts_and_full_validation(self):
        report = self.run_split()
        self.assertEqual(report['distribution']['R31'], {'total':29, 'train':20, 'val':3, 'test':6})
        self.assertTrue(validate_dataset(self.root)['success'])

    def test_deterministic_rerun(self):
        self.run_split()
        before = [(self.root/f'{n}.jsonl').read_bytes() for n in ('train','val','test')]
        self.run_split()
        self.assertEqual(before, [(self.root/f'{n}.jsonl').read_bytes() for n in ('train','val','test')])

    def test_balancing_changes_train_only_and_weights_use_train(self):
        self.run_split()
        before = [(self.root/f'{n}.jsonl').read_bytes() for n in ('val','test')]
        report = self.run_split(balance='oversample')
        self.assertEqual(report['train_count'],105)
        self.assertEqual(before, [(self.root/f'{n}.jsonl').read_bytes() for n in ('val','test')])
        self.assertTrue(validate_dataset(self.root)['success'])
        self.run_split(balance='class_weight')
        labels = json.loads((self.root/'labels.json').read_text(encoding='utf-8'))
        self.assertAlmostEqual(labels['class_weights']['R31'], 90/(3*20))

    def test_invalid_ratio_does_not_write(self):
        with self.assertRaises(ValueError):
            self.run_split(test_size=0.8, val_size=0.3)
        self.assertFalse((self.root/'train.jsonl').exists())

    def test_duplicate_id_or_text_is_rejected(self):
        rows = read_rows(self.cleaned)
        write_rows(self.cleaned, rows + [rows[0]])
        with self.assertRaises(ValueError):
            self.run_split()

    def test_validator_detects_tampered_labels(self):
        self.run_split()
        path = self.root/'test.jsonl'
        rows = read_rows(path)
        rows[0]['label_id'] = 99
        write_rows(path, rows)
        with self.assertRaises(ValueError):
            validate_dataset(self.root)


class ImportTest(unittest.TestCase):
    def test_identical_packages_skip_and_old_file_is_backed_up(self):
        from preprocess.import_raw import import_raw
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            row = {'title':'新版', 'clc_code':'R51'}
            for folder in ('first','second'):
                write_rows(root/f'{folder}/raw/R51.jsonl', [row])
            write_rows(root/'data/raw/R51.jsonl', [dict(row, title='旧版')])
            with patch('preprocess.import_raw.DATA_DIR', root/'data'), \
                 patch('preprocess.import_raw.RAW_DIR', root/'data/raw'), \
                 patch('preprocess.import_raw.PROJECT_ROOT', root):
                report = import_raw([root/'first', root/'second'], replace_existing=True)
            self.assertEqual(report['identical_category_files_skipped'], 1)
            backup = Path(report['backup_directory'])/'raw/R51.jsonl'
            self.assertEqual(read_rows(backup)[0]['title'], '旧版')
            self.assertEqual(read_rows(root/'data/raw/R51.jsonl')[0]['title'], '新版')

    def test_different_incoming_versions_fail_before_copy(self):
        from preprocess.import_raw import import_raw
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for folder in ('first','second'):
                write_rows(root/f'{folder}/raw/R51.jsonl', [{'title':folder, 'clc_code':'R51'}])
            with patch('preprocess.import_raw.DATA_DIR', root/'data'), \
                 patch('preprocess.import_raw.RAW_DIR', root/'data/raw'):
                with self.assertRaises(ValueError):
                    import_raw([root/'first', root/'second'], replace_existing=True)
            self.assertFalse((root/'data').exists())

    def test_clean_raw_deduplicates_and_keeps_richer_record(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw_dir = root / "raw"
            output_dir = root / "processed"
            raw_dir.mkdir()
            base = {"title": "同一篇论文", "source": "测试期刊", "year": "2026-08-20",
                    "clc_code": "R51", "uid": "old", "abstract": "", "keywords": ""}
            rich = dict(base, year="2026", uid="new", doi="10.1/test",
                        abstract="完整摘要", keywords="甲 ; 乙")
            with (raw_dir / "R51.jsonl").open("w", encoding="utf-8") as handle:
                handle.write(json.dumps(base, ensure_ascii=False) + "\n")
                handle.write(json.dumps(rich, ensure_ascii=False) + "\n")
            report = clean_raw(raw_dir, output_dir)
            self.assertEqual(report["raw_count"], 2)
            self.assertEqual(report["cleaned_count"], 1)
            self.assertEqual(report["duplicates_removed"], 1)
            row = json.loads((output_dir / "cleaned.jsonl").read_text(encoding="utf-8"))
            self.assertEqual(row["abstract"], "完整摘要")
            self.assertEqual(row["keywords"], ["甲", "乙"])


if __name__ == "__main__":
    unittest.main()
