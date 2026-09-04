"""Offline orchestration of member 1-3 scripts; online inference never imports training."""

import argparse
import json
import sys
from pathlib import Path
from time import perf_counter
from typing import Literal

from pydantic import Field, model_validator

from agent.adapters import require_checkpoint, run_json_command
from agent.errors import AgentError
from agent.factory import resolve_path
from agent.schemas import Schema


class StageSettings(Schema):
    command: list[str] = Field(min_length=1)
    cwd: str = "."
    outputs: list[str] = Field(min_length=1)
    timeout_seconds: float = Field(default=3600, gt=0)

    @model_validator(mode="after")
    def validate_command(self):
        if any(not value.strip() for value in self.command):
            raise ValueError("command arguments must not be empty")
        return self


class PipelineSettings(Schema):
    checkpoint_path: str
    stages: dict[Literal["collect", "clean", "train"], StageSettings] = Field(default_factory=dict)


def run_pipeline(
    settings: PipelineSettings,
    *,
    base: Path,
    prepare_data: bool = False,
    train_if_missing: bool = False,
) -> dict:
    """Only explicitly requested stages run; existing weights always skip training."""
    steps = []
    started = perf_counter()
    checkpoint = resolve_path(base, settings.checkpoint_path)
    active_stage = "validate_pipeline"
    try:
        try:
            require_checkpoint(checkpoint)
            has_checkpoint = True
        except AgentError:
            has_checkpoint = False
        stages_to_run = ["collect", "clean"] if prepare_data else []
        if not has_checkpoint:
            if not train_if_missing:
                raise AgentError(
                    "MODEL_NOT_READY", "没有已有权重；如需训练，请显式使用 --train-if-missing。"
                )
            stages_to_run.append("train")
        for name in stages_to_run:
            if name not in settings.stages:
                raise AgentError("CONFIG_ERROR", f"未配置 {name} 阶段的脚本。")

        for name in ("collect", "clean", "train"):
            active_stage = name
            if name not in stages_to_run:
                message = "已有权重，跳过训练" if name == "train" else "未请求数据准备，跳过"
                steps.append(
                    {"step": name, "status": "skipped", "message": message, "duration_ms": 0}
                )
                continue
            stage = settings.stages[name]
            stage_started = perf_counter()
            output = run_json_command(
                [sys.executable if arg == "{python}" else arg for arg in stage.command],
                {"action": name, "checkpoint_path": str(checkpoint)},
                cwd=resolve_path(base, stage.cwd),
                timeout=stage.timeout_seconds,
            )
            if output.get("status") != "success":
                raise AgentError("PIPELINE_STAGE_FAILED", f"{name} 阶段未报告成功。")
            for artifact in stage.outputs:
                artifact_path = resolve_path(base, artifact)
                if (
                    not artifact_path.exists()
                    or (artifact_path.is_file() and artifact_path.stat().st_size == 0)
                    or (artifact_path.is_dir() and not any(artifact_path.iterdir()))
                ):
                    raise AgentError("MISSING_ARTIFACT", f"{name} 阶段缺少有效的约定输出文件。")
            if name == "train":
                require_checkpoint(checkpoint)
            steps.append(
                {
                    "step": name,
                    "status": "success",
                    "message": f"{name} 阶段及输出检查完成",
                    "duration_ms": round((perf_counter() - stage_started) * 1000, 3),
                }
            )
        return {
            "status": "success",
            "checkpoint_path": str(checkpoint),
            "steps": steps,
            "duration_ms": round((perf_counter() - started) * 1000, 3),
            "error": None,
        }
    except AgentError as exc:
        steps.append({"step": active_stage, "status": "error", "message": exc.message})
        return {
            "status": "error",
            "checkpoint_path": str(checkpoint),
            "steps": steps,
            "duration_ms": round((perf_counter() - started) * 1000, 3),
            "error": {"code": exc.code, "message": exc.message},
        }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="离线采集、清洗、训练脚本编排")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--prepare-data", action="store_true", help="执行采集和清洗脚本")
    parser.add_argument("--train-if-missing", action="store_true", help="缺少权重时执行训练脚本")
    args = parser.parse_args(argv)
    try:
        path = args.config.resolve()
        settings = PipelineSettings.model_validate_json(path.read_text(encoding="utf-8"))
        result = run_pipeline(
            settings,
            base=path.parent,
            prepare_data=args.prepare_data,
            train_if_missing=args.train_if_missing,
        )
    except (OSError, ValueError) as exc:
        result = {
            "status": "error",
            "error": {
                "code": "CONFIG_ERROR",
                "message": f"无法读取离线流程配置（{type(exc).__name__}）。",
            },
        }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
