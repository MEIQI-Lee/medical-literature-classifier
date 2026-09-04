"""Explicit offline tool runner. With no command, only inspect the deployment bundle."""

import argparse
import json

from agent.integrations.medbert_backend import inspect_bundle
from agent.integrations.team_contract import failure_result
from agent.tools import get_tool
from agent.tools.base import ToolResult


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="离线工具入口；默认只检查已有权重")
    commands = parser.add_subparsers(dest="task")
    commands.add_parser("check", help="检查已有模型，不训练")
    collect = commands.add_parser("collect", help="显式启动数据采集")
    collect.add_argument("--categories", nargs="+", required=True)
    collect.add_argument("--target-count", type=int, default=3000)
    commands.add_parser("clean", help="显式重新构建数据集")
    train = commands.add_parser("train", help="显式训练新版本；不替换当前部署模型")
    train.add_argument("--output-name", required=True)
    train.add_argument("--epochs", type=int, default=20)
    train.add_argument("--device")
    args = parser.parse_args(argv)
    try:
        if args.task in (None, "check"):
            bundle = inspect_bundle()
            result = ToolResult.ok(
                {
                    "deployment": {key: value for key, value in bundle.items() if key != "labels"},
                    "training_skipped": True,
                }
            )
        elif args.task == "collect":
            result = get_tool("collect_data").run(
                categories=args.categories,
                target_count=args.target_count,
            )
        elif args.task == "clean":
            result = get_tool("clean_dataset").run()
        else:
            result = get_tool("train_model").run(
                model_type="medbert",
                output_name=args.output_name,
                epochs=args.epochs,
                device=args.device,
            )
    except Exception as exc:
        result = failure_result(exc)
    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
    return 0 if result.success else 1


if __name__ == "__main__":
    raise SystemExit(main())
