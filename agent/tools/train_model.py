"""Agent Tool：训练并评估文本分类模型。"""

from __future__ import annotations

from datetime import datetime
import re

from agent.tools.base import BaseTool, ToolResult
from config import CHECKPOINT_DIR


class TrainModelTool(BaseTool):
    """训练新模型版本，并在默认测试集上完成评估。"""

    name = "train_model"
    description = (
        "使用本地 MedBERT 预训练底座训练学术文本分类模型，并完成测试集评估。"
        "默认训练 MedBERT 20 轮；每次会生成新版本，不覆盖当前 medbert_v20 部署模型。"
    )
    input_schema = {
        "type": "object",
        "properties": {
            "model_type": {
                "type": "string",
                "enum": ["medbert", "macbert", "qwen_embedding"],
                "default": "medbert",
                "description": "训练模型类型；默认 medbert",
            },
            "epochs": {
                "type": "integer",
                "default": 20,
                "minimum": 1,
                "description": "训练轮数",
            },
            "output_name": {
                "type": "string",
                "description": "新 checkpoint 名称；省略时自动生成",
            },
            "device": {
                "type": "string",
                "description": "可选，例如 cuda 或 cpu；省略时自动选择",
            },
        },
    }

    @staticmethod
    def _new_run_name(model_type: str) -> str:
        return f"{model_type}_{datetime.now():%Y%m%d_%H%M%S}"

    @staticmethod
    def _validate_run_name(output_name: str) -> None:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", output_name):
            raise ValueError("output_name 只能包含字母、数字、下划线和连字符")
        if output_name == "medbert_v20":
            raise ValueError("禁止覆盖当前默认部署模型 medbert_v20；请使用新的 output_name")
        if (CHECKPOINT_DIR / output_name).exists():
            raise FileExistsError(f"checkpoint 已存在，拒绝覆盖: {CHECKPOINT_DIR / output_name}")

    def run(self, model_type: str = "medbert", epochs: int = 20,
            output_name: str | None = None, device: str | None = None,
            train_file: str | None = None, val_file: str | None = None,
            test_file: str | None = None, **kwargs) -> ToolResult:
        """训练新版本并评估；异常统一封装为 ToolResult.fail。"""
        self.steps = []
        self.logs = []
        train_step = None
        try:
            if model_type not in {"medbert", "macbert", "qwen_embedding"}:
                raise ValueError("model_type 必须是 medbert、macbert 或 qwen_embedding")
            if not isinstance(epochs, int) or epochs < 1:
                raise ValueError("epochs 必须是大于 0 的整数")

            run_name = output_name or self._new_run_name(model_type)
            self._validate_run_name(run_name)
            self.log(f"训练版本: {run_name}")

            from model.train import train_model
            train_step = self.add_step("train_model", f"训练 {model_type}，epochs={epochs}")
            train_args = {
                "model_type": model_type,
                "epochs": epochs,
                "output_name": run_name,
                "device": device,
            }
            if train_file:
                train_args["train_file"] = train_file
            if val_file:
                train_args["val_file"] = val_file
            training = train_model(**train_args)
            best = training.get("best_metrics") or {}
            self.update_step(train_step, "done", f"checkpoint={training['checkpoint']}，val_macro_f1={best.get('f1_macro')}")

            from model.evaluate import evaluate_model
            evaluate_step = self.add_step("evaluate_model", "在测试集上评估最佳 checkpoint")
            evaluate_args = {
                "model_type": model_type,
                "checkpoint": training["checkpoint"],
                "device": device,
                "output_name": f"{run_name}_test",
            }
            if test_file:
                evaluate_args["test_file"] = test_file
            evaluation = evaluate_model(**evaluate_args)
            self.update_step(evaluate_step, "done", f"accuracy={evaluation['accuracy']}，macro_f1={evaluation['f1_macro']}")
            self.log(f"测试指标文件: {evaluation['metrics_path']}")
            return ToolResult.ok(
                data={"run_name": run_name, "training": training, "evaluation": evaluation},
                steps=self.steps,
                logs=self.logs,
            )
        except Exception as exc:
            if self.steps:
                self.update_step(self.steps[-1], "failed", str(exc))
            self.log(f"训练或评估失败: {exc}")
            return ToolResult.fail(error=str(exc), steps=self.steps, logs=self.logs)
