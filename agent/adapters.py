import importlib
import json
import logging
import os
import subprocess
from pathlib import Path
from threading import Lock
from typing import Any, Protocol

from pydantic import ValidationError

from agent.errors import AgentError
from agent.schemas import ModelInfo, ModelOutput

logger = logging.getLogger(__name__)


class Classifier(Protocol):
    def describe(self) -> ModelInfo: ...

    def predict(self, text: str, top_k: int = 5) -> ModelOutput: ...


def load_symbol(reference: str):
    """Only load trusted deployment configuration, never an uploaded document's text."""
    try:
        module, attribute = reference.split(":", 1)
        return getattr(importlib.import_module(module), attribute)
    except (ValueError, ImportError, AttributeError) as exc:
        raise AgentError("CONFIG_ERROR", "无法导入配置中的 Python 工厂函数。") from exc


def require_checkpoint(path: str | Path) -> Path:
    checkpoint = Path(path).resolve()
    if (
        not checkpoint.exists()
        or (checkpoint.is_file() and checkpoint.stat().st_size == 0)
        or (checkpoint.is_dir() and not any(checkpoint.iterdir()))
    ):
        raise AgentError("MODEL_NOT_READY", "未找到已有模型权重，请先通过离线入口完成训练。")
    return checkpoint


def run_json_command(
    command: list[str], payload: dict, *, cwd: Path, timeout: float
) -> dict[str, Any]:
    """JSON over stdin/stdout. Commands are argv lists and never evaluated by a shell."""
    try:
        result = subprocess.run(
            command,
            input=json.dumps(payload, ensure_ascii=False, allow_nan=False),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="strict",
            cwd=cwd,
            timeout=timeout,
            check=False,
            env={**os.environ, "PYTHONIOENCODING": "utf-8"},
        )
    except subprocess.TimeoutExpired as exc:
        raise AgentError("TOOL_TIMEOUT", "脚本执行超时。") from exc
    except (OSError, UnicodeError) as exc:
        raise AgentError("TOOL_EXECUTION_FAILED", "无法启动脚本或读取脚本输出。") from exc
    if result.returncode:
        # Do not copy arbitrary stderr (which may contain input text) into API responses.
        logger.error("Configured script exited with status %s", result.returncode)
        raise AgentError("TOOL_EXECUTION_FAILED", f"脚本执行失败，退出码为 {result.returncode}。")
    try:
        output = json.loads(result.stdout)
        if not isinstance(output, dict):
            raise ValueError("expected one JSON object")
        return output
    except (ValueError, TypeError) as exc:
        raise AgentError("INVALID_TOOL_OUTPUT", "脚本 stdout 必须只输出一个 JSON 对象。") from exc


class PythonModelAdapter:
    """The backend is created once; inference is serialized for simple GPU integration."""

    def __init__(self, backend: Any):
        self.backend = backend
        self._lock = Lock()
        try:
            self._info = ModelInfo.model_validate(backend.describe())
        except Exception as exc:
            raise AgentError("MODEL_NOT_READY", "无法读取模型信息，请检查模型工厂及权重。") from exc

    def describe(self) -> ModelInfo:
        return self._info.model_copy(deep=True)

    def predict(self, text: str, top_k: int = 5) -> ModelOutput:
        try:
            with self._lock:
                output = self.backend.predict(text, top_k=top_k)
            return ModelOutput.model_validate(output)
        except AgentError:
            raise
        except ValidationError as exc:
            raise AgentError("INVALID_MODEL_OUTPUT", "模型输出不符合约定的数据结构。") from exc
        except Exception as exc:
            logger.error("Model inference raised %s", type(exc).__name__)
            raise AgentError("MODEL_INFERENCE_FAILED", "模型推理失败，请检查模型实现。") from exc


class ScriptModelAdapter:
    """Compatibility with an existing one-shot script (one new process per request)."""

    def __init__(
        self, command: list[str], checkpoint_path: Path, *, cwd: Path, timeout: float = 60
    ):
        if not command or any(not isinstance(arg, str) or not arg for arg in command):
            raise AgentError("CONFIG_ERROR", "脚本命令必须是非空字符串参数列表。")
        self.command = list(command)
        self.checkpoint_path = require_checkpoint(checkpoint_path)
        self.cwd = cwd
        self.timeout = timeout
        self._lock = Lock()
        try:
            self._info = ModelInfo.model_validate(self._call("describe"))
        except ValidationError as exc:
            raise AgentError(
                "INVALID_MODEL_OUTPUT", "脚本 describe 输出不符合模型信息协议。"
            ) from exc

    def _call(self, action: str, **kwargs) -> dict:
        with self._lock:
            return run_json_command(
                self.command,
                {"action": action, "checkpoint_path": str(self.checkpoint_path), **kwargs},
                cwd=self.cwd,
                timeout=self.timeout,
            )

    def describe(self) -> ModelInfo:
        return self._info.model_copy(deep=True)

    def predict(self, text: str, top_k: int = 5) -> ModelOutput:
        try:
            return ModelOutput.model_validate(self._call("predict", text=text, top_k=top_k))
        except ValidationError as exc:
            raise AgentError("INVALID_MODEL_OUTPUT", "脚本 predict 输出不符合推理协议。") from exc
