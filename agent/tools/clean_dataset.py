"""
Tool 2: 数据清洗与数据集构建 (成员2负责)
================================
职责: 去重/缺失值/噪声清理/标签映射/数据集划分,输出 train/val/test.jsonl + labels.json
"""
from agent.tools.base import BaseTool, ToolResult


class CleanDatasetTool(BaseTool):
    """数据清洗与数据集构建工具"""

    name = "clean_dataset"
    description = (
        "对原始文献数据进行去重、缺失值处理、HTML/特殊字符清洗、"
        "字段统一格式化、分类号→标准标签映射、构造'标题+关键词+摘要'文本输入,"
        "并划分训练集/验证集/测试集。输出 train.jsonl/val.jsonl/test.jsonl + labels.json。"
    )
    input_schema = {
        "type": "object",
        "properties": {
            "raw_dir": {
                "type": "string",
                "description": "原始数据目录,默认 data/raw/",
                "default": "data/raw"
            },
            "test_size": {
                "type": "number",
                "description": "测试集比例,默认0.2",
                "default": 0.2
            },
            "val_size": {
                "type": "number",
                "description": "验证集比例,默认0.1",
                "default": 0.1
            },
            "balance": {
                "type": "string",
                "enum": ["none", "oversample", "class_weight"],
                "description": "类别不平衡处理方式",
                "default": "none"
            },
            "random_state": {
                "type": "integer", "default": 42, "description": "确定性划分随机种子"
            }
        }
    }

    def run(self, raw_dir: str = None, test_size: float = 0.2,
            val_size: float = 0.1, balance: str = "none", random_state: int = 42,
            **kwargs) -> ToolResult:
        from config import RAW_DIR
        from preprocess.clean import clean_raw
        from preprocess.split import split_dataset
        from preprocess.statistics import compute_statistics
        from preprocess.validate import validate_dataset

        self.steps = []
        self.logs = []
        try:
            if not 0 <= test_size < 1 or not 0 <= val_size < 1 or test_size + val_size >= 1:
                raise ValueError("划分比例不合法，不修改已有数据集")
            if balance not in {'none', 'oversample', 'class_weight'}:
                raise ValueError("未知 balance，不修改已有数据集")
            clean_step = self.add_step("clean_raw", "清洗、校验并去重原始数据")
            clean_report = clean_raw(raw_dir=raw_dir or RAW_DIR)
            self.update_step(clean_step, "done", f"保留 {clean_report['cleaned_count']} 条")
            self.log(f"删除重复记录 {clean_report['duplicates_removed']} 条")
            self.log(f"隔离 {clean_report['label_conflict_records']} 条标签冲突记录")
            if clean_report['excluded_categories']:
                self.log(f"无可训练样本的类别: {clean_report['excluded_categories']}")

            statistics_step = self.add_step("compute_statistics", "生成数据质量与分布统计")
            statistics_report = compute_statistics()
            self.update_step(statistics_step, "done", f"统计 {statistics_report['label_count']} 个类别")

            split_step = self.add_step("split_dataset", "按类别分层划分数据集")
            split_report = split_dataset(test_size=test_size, val_size=val_size,
                                         balance=balance, random_state=random_state)
            self.update_step(
                split_step, "done",
                f"train={split_report['train_count']}, val={split_report['val_count']}, test={split_report['test_count']}",
            )
            validate_step = self.add_step('validate_dataset', '核对数据数量、标签版本和跨集合泄漏')
            validation_report = validate_dataset()
            self.update_step(validate_step, 'done', '全部结构及泄漏校验通过')
            return ToolResult.ok(
                data={"cleaning": clean_report, "statistics": statistics_report,
                      "split": split_report, "validation": validation_report},
                steps=self.steps, logs=self.logs,
            )
        except Exception as exc:
            if self.steps:
                self.update_step(self.steps[-1], "failed", str(exc))
            self.log(f"数据集构建失败: {exc}")
            return ToolResult.fail(error=str(exc), steps=self.steps, logs=self.logs)
