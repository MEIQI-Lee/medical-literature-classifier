"""Executable service example; intentionally does not implement member 5's web routes.

Run from project root: python -m examples.member5_client [--demo] [--file paper.pdf]
"""

import argparse
import json
import sys
from pathlib import Path

from agent import AcademicClassificationAgent, create_demo_agent, get_agent


def main():
    parser = argparse.ArgumentParser(description="成员5：Agent 服务调用示例")
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--file", type=Path)
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()
    service = AcademicClassificationAgent(core=create_demo_agent()) if args.demo else get_agent()

    def on_step(event):
        # Member 5 may forward these events to SSE/WebSocket; callback runs synchronously.
        print(json.dumps(event, ensure_ascii=False), file=sys.stderr)

    if args.file:
        with args.file.open("rb") as handle:
            content = handle.read(10 * 1024 * 1024 + 1)
        result = service.classify_file(
            content,
            args.file.name,
            top_k=args.top_k,
            on_step=on_step,
        )
    else:
        result = service.classify_paper(
            title="高血压患者心血管危险因素分析",
            keywords=["高血压", "心血管疾病"],
            abstract="分析高血压患者的心血管危险因素，比较不同治疗方案的临床效果。",
            top_k=args.top_k,
            on_step=on_step,
        )
    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
    return 0 if result.success else 1


if __name__ == "__main__":
    raise SystemExit(main())
