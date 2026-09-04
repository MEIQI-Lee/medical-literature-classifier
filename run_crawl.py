"""
入口脚本: 遍历 128 个 CLC 三级类目采集知网文献
================================================

用法:
  # 快速测试: 只爬 R51, 5 条, 有头浏览器, 含详情页提取
  python run_crawl.py --test

  # 仅表格基础字段 (不打开详情页, 速度快)
  python run_crawl.py --no-detail --limit 30

  # 指定部分类目 + 详情页提取 (17 字段完整)
  python run_crawl.py --cats R51 R52 R54 --limit 50

  # 全部 128 类目 + 详情页提取
  python run_crawl.py --limit 30

输出:
  data/raw/R51.jsonl, data/raw/R52.jsonl, ...  (每类一个 jsonl)
  data/raw/cnki_papers.csv                     (全部合并的 CSV)
"""
import argparse
import json
import sys
from pathlib import Path

# 确保项目根在 sys.path
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from config import RAW_DIR, CATEGORY_MAPPING_FILE  # noqa: E402
from crawler.cnki import run_full_crawl  # noqa: E402
from loguru import logger  # noqa: E402


def main():
    parser = argparse.ArgumentParser(
        description="按 CLC 三级类目采集知网文献",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--test", action="store_true",
        help="快速测试: 只跑 R51, 20 条, 有头浏览器",
    )
    parser.add_argument(
        "--cats", nargs="+", default=None,
        help="指定类目编码列表 (如 --cats R51 R52 R54)",
    )
    parser.add_argument(
        "--limit", type=int, default=30,
        help="每个类目最多采集多少条 (默认 30)",
    )
    parser.add_argument(
        "--headless", action="store_true",
        help="无头浏览器模式",
    )
    parser.add_argument(
        "--no-split", action="store_true",
        help="不按类目分文件, 合并写一个 cnki_papers.jsonl",
    )
    parser.add_argument(
        "--no-detail", action="store_true",
        help="跳过详情页提取 (仅表格基础字段, 速度快)",
    )
    parser.add_argument(
        "--list", action="store_true",
        help="列出所有类目及其名称",
    )
    args = parser.parse_args()

    # 列出类目
    if args.list:
        mapping = json.loads(CATEGORY_MAPPING_FILE.read_text(encoding="utf-8"))
        print(f"共 {len(mapping)} 个三级类目:\n")
        for code, name in sorted(mapping.items()):
            print(f"  {code}\t{name}")
        return

    # 测试模式: R51, 5 条, 含详情页
    if args.test:
        logger.info("=== 快速测试模式: R51, 5 条, 含详情页 ===")
        result = run_full_crawl(
            max_per_category=5,
            categories=["R51"],
            resume=False,
            headless=False,
            save_per_category=True,
            fetch_detail=True,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    # 正式运行
    categories = args.cats
    if categories is None:
        mapping = json.loads(CATEGORY_MAPPING_FILE.read_text(encoding="utf-8"))
        categories = sorted(mapping.keys())
        logger.info("将采集全部 {} 个三级类目", len(categories))
    else:
        logger.info("指定采集 {} 个类目: {}", len(categories), categories)

    total_limit = len(categories) * args.limit
    logger.info("每类 {} 条, 总计上限 {} 条", args.limit, total_limit)
    logger.info("输出目录: {}", RAW_DIR)

    # 询问确认 (非交互跳过)
    result = run_full_crawl(
        max_per_category=args.limit,
        categories=categories,
        headless=args.headless,
        save_per_category=not args.no_split,
        fetch_detail=not args.no_detail,
    )
    print("\n" + "=" * 50)
    print("采集完成!")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
