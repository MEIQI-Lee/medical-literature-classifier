import argparse
import json
import sys
from pathlib import Path

from agent.errors import AgentError
from agent.factory import create_agent, create_demo_agent, create_medbert_agent
from agent.tracing import JsonlTraceSink


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="学术文本分类 Agent（成员 4）")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--demo", action="store_true", help="明确使用模拟分类器联调")
    mode.add_argument("--config", type=Path, help="真实模型的 JSON 配置")
    mode.add_argument("--team", action="store_true", help="使用成员3的本地 MedBERT v20 权重")
    parser.add_argument("--checkpoint", type=Path, help="--team 的 checkpoint 目录")
    parser.add_argument("--device", help="--team 的推理设备，例如 cpu 或 cuda")
    parser.add_argument("--top-k", type=int, help="本次请求的候选数；低置信度至少返回3项")
    parser.add_argument("--confidence-threshold", type=float, help="本次请求的概率阈值")
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--text", help="直接输入文本")
    inputs.add_argument("--file", type=Path, help="TXT / PDF / DOCX 文件")
    inputs.add_argument("--stdin", action="store_true", help="从标准输入读取文本")
    inputs.add_argument("--paper-json", type=Path, help="包含 title/keywords/abstract 的 JSON 文件")
    inputs.add_argument("--graph", action="store_true", help="输出 Mermaid 工作流图")
    parser.add_argument("--steps", action="store_true", help="将实时步骤 JSON 输出到 stderr")
    parser.add_argument("--trace-file", type=Path, help="将最终结果追加保存到 JSONL 日志")
    args = parser.parse_args(argv)
    try:
        if args.team:
            agent = create_medbert_agent(checkpoint_path=args.checkpoint, device=args.device)
        else:
            agent = create_demo_agent() if args.demo else create_agent(args.config)
        if args.trace_file:
            agent.trace_sink = JsonlTraceSink(args.trace_file)
        if args.graph:
            print(agent.graph.get_graph().draw_mermaid())
            return 0

        def on_step(step):
            print(step.model_dump_json(), file=sys.stderr, flush=True)

        callback = on_step if args.steps else None
        options = {
            "on_step": callback,
            "top_k": args.top_k,
            "confidence_threshold": args.confidence_threshold,
        }
        if args.text is not None:
            result = agent.classify_text(args.text, **options)
        elif args.file:
            with args.file.open("rb") as handle:
                content = handle.read(agent.config.max_file_bytes + 1)
            result = agent.classify_file(content, args.file.name, **options)
        elif args.paper_json:
            payload = json.loads(args.paper_json.read_text(encoding="utf-8"))
            if not isinstance(payload, dict) or set(payload) - {"title", "keywords", "abstract"}:
                raise AgentError(
                    "INVALID_INPUT", "论文 JSON 仅接受 title、keywords、abstract 字段。"
                )
            result = agent.classify_paper(**payload, **options)
        else:
            result = agent.classify_text(sys.stdin.read(agent.config.max_text_chars + 1), **options)
        print(result.model_dump_json(indent=2))
        return 0 if result.status == "success" else 1
    except (AgentError, OSError, ValueError) as exc:
        error = (
            {"code": exc.code, "message": exc.message}
            if isinstance(exc, AgentError)
            else {
                "code": "INPUT_OR_CONFIG_ERROR",
                "message": "无法读取输入或配置文件，请检查路径和内容。",
            }
        )
        print(json.dumps({"status": "error", "error": error}, ensure_ascii=False))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
