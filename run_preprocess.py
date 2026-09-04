"""成员2数据清洗与数据集构建的一键运行入口。"""

import argparse
import json
import sys

from config import RAW_DIR
from agent.tools.clean_dataset import CleanDatasetTool


def main() -> None:
    parser = argparse.ArgumentParser(description="清洗原始文献并生成训练/验证/测试集")
    parser.add_argument("--raw-dir", default=str(RAW_DIR), help="原始 R*.jsonl 所在目录")
    parser.add_argument("--test-size", type=float, default=0.2, help="测试集占总数据比例")
    parser.add_argument("--val-size", type=float, default=0.1, help="验证集占总数据比例")
    parser.add_argument("--balance", choices=["none", "oversample", "class_weight"], default="none")
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    result = CleanDatasetTool().run(
        raw_dir=args.raw_dir, test_size=args.test_size,
        val_size=args.val_size, balance=args.balance, random_state=args.random_state,
    )
    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
    if not result.success:
        raise SystemExit(1)


if __name__ == "__main__":
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    main()
