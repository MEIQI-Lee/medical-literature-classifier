# 成员2：数据清洗与数据集构建

当前批次（2026-09-03）：原始128类6379条，清洗后124类6162条；完整交接见
`../MEMBER2_HANDOFF.md` 和 `DATASET_REPORT_2026-09-03.md`。
仅使用Python标准库，建议使用项目现有Python 3.11虚拟环境。

## 一键运行

在项目根目录执行：

```bash
python run_preprocess.py
python -m preprocess.validate
python -m unittest discover -s tests -v
```

默认只读取 `data/raw/R*.jsonl`，不会再次读取内容相同的合并文件
`data/raw/cnki_papers.csv`。默认按 70%/10%/20% 划分训练集、验证集和测试集。

可选参数：

```bash
python run_preprocess.py --raw-dir data/raw --val-size 0.1 --test-size 0.2 --balance none
```

`balance` 支持：

- `none`：不修改训练集分布；
- `class_weight`：不复制数据，在 `data/labels.json` 中提供类别权重；
- `oversample`：只对训练集随机过采样，不改变验证集和测试集。

## 文件职责

- `clean.py`：字段清洗、HTML/Unicode处理、年份统一、关键词拆分、标签校验、去重、质量标记。
- `split.py`：确定性分层划分、标签编号、类别权重和可选训练集过采样。
- `statistics.py`：类别分布、缺失率、文本长度、质量标记和不平衡比例。
- `../agent/tools/clean_dataset.py`：把以上三个步骤包装成 Agent 可调用工具。
- `../run_preprocess.py`：命令行一键入口。
- `import_raw.py`：核验交付包、跳过完全相同文件、替换前归档旧数据，保存导入清单。
- `validate.py`：自动校验源数据指纹、训练标签版本、数量守恒及跨集合泄漏。

## 输出文件

- `data/processed/cleaned.jsonl`：去重后的标准数据集。
- `data/processed/rejected.jsonl`：JSON错误、标题缺失、未知标签或标签冲突记录。
- `data/processed/quality_report.json`：清洗前后数量和主要质量问题。
- `data/processed/statistics.json`：类别、文本长度和缺失字段统计。
- `data/processed/split_report.json`：划分参数及各类别划分数量。
- `data/train.jsonl`、`data/val.jsonl`、`data/test.jsonl`：成员3直接使用的数据集。
- `data/labels.json`：`label2id`、`id2label`、类别名称和训练类别权重。
- `data/processed/label_conflicts.jsonl`：冲突组复核清单，不自动推断主标签。
- `data/processed/duplicate_groups.jsonl`：同标签合并审计及来源。
- `data/processed/validation_report.json`：结构、版本与泄漏验收结果。
- `data/import_manifest.json`：导入来源、SHA-256及备份位置。

## 清洗约定

1. DOI相同或“标准化标题 + 标准化来源”相同时归为同组；完全相同模型输入也归为同组（如转载），避免重复文本泄漏。这是训练数据等价合并，不意味着能证实不同来源在书目意义上完全相同。
2. 重复记录保留字段最完整的一条，并用同组其他记录补空字段。
3. 年份统一为整数，例如 `2026-08-20 10:00` 转为 `2026`，原值保存在 `year_raw`。
4. 摘要缺失或可能以省略号截断时不伪造内容，通过 `quality_flags` 标记。
5. 当前标签来自采集检索条件 `clc_code`，不是详情页真实 `classification`，属于弱标签。
6. 分割在去重之后执行，防止同一论文进入不同集合造成数据泄漏。
7. 同组标签冲突时全部隔离；`rejected_count`按原始记录计数，`duplicates_removed`仅统计非冲突组的冗余条数。
8. `cleaned_count + rejected_count + duplicates_removed = raw_count`；详细计数见quality_report。
9. 只删除明确HTML标签，保留P值比较符号和希腊字母；支持202304这样的紧凑年月日期。
10. 全流程使用固定随机种子42；相同输入和代码下结果可复现，追加新数据则会重建划分，需要重新训练。

## 导入新交付包

```bash
python -m preprocess.import_raw --source "交付包1目录" "交付包2目录" --replace-existing
python run_preprocess.py
```

目录应为 `raw` 本身或包含 `raw` 的上级目录。只复制分类JSONL和合并CSV。
先验证CSV与JSONL逐字段一致；交付包之间的同名不同内容会停止导入，不能盲选版本。
`--replace-existing` 允许用成员1新版替换项目同名旧文件；替换前备份旧raw、processed、划分及文档至data/archive。

## 训练数据接口

每条训练样本包含：`id: str`、`text: str`、`label: str`、`label_id: int`、
`title: str`、`keywords: list[str]`、`abstract: str`、`year: int|null`、`year_raw: str`，
以及`source`、`doi`、`url`、`quality_flags`、`provenance`等追溯字段。
训练只输入`text`并读取`label_id`；不得把标签、来源、质量标记、原始文件名拼进模型输入。
`label2id`只包含清洗后确有样本的类别；分类头、推理代码与训练数据必须共用当前labels文件。
`dataset_sha256`绑定cleaned文件，出现在标签/清洗/统计/划分/验收报告中；是数据版本校验值，不是模型权重。

文本长度统计单位为Unicode字符，不是token。预训练模型应由成员3使用对应tokenizer计算截断情况，不能把512字符直接当512 token。
