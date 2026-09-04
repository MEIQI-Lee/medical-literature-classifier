import sys
from importlib.resources import files
from pathlib import Path
from typing import Literal

from pydantic import Field, ValidationError, model_validator

from agent.adapters import (
    PythonModelAdapter,
    ScriptModelAdapter,
    load_symbol,
    require_checkpoint,
)
from agent.agent import ClassificationAgent
from agent.catalog import CategoryCatalog
from agent.demo import DemoClassifier
from agent.errors import AgentError
from agent.schemas import AgentConfig, ConfidencePolicy, Schema
from agent.tools.text import DefaultTextPreprocessor
from agent.tracing import JsonlTraceSink


class ModelSettings(Schema):
    kind: Literal["python", "script"]
    checkpoint_path: str
    factory: str | None = None
    command: list[str] | None = None
    cwd: str = "."
    timeout_seconds: float = Field(default=60, gt=0)

    @model_validator(mode="after")
    def validate_entrypoint(self):
        if self.kind == "python" and (not self.factory or self.command is not None):
            raise ValueError("python kind requires factory and no command")
        if self.kind == "script" and (not self.command or self.factory is not None):
            raise ValueError("script kind requires command and no factory")
        return self


class RuntimeSettings(Schema):
    labels_path: str
    categories_path: str
    model: ModelSettings
    preprocessor_factory: str | None = None
    agent: AgentConfig = Field(default_factory=AgentConfig)
    trace_path: str | None = None


def resolve_path(base: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else base / path).resolve()


def create_agent(config_path: str | Path) -> ClassificationAgent:
    """Load existing weights only. This function never invokes training."""
    path = Path(config_path).resolve()
    try:
        settings = RuntimeSettings.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, ValidationError) as exc:
        raise AgentError("CONFIG_ERROR", "无法读取 Agent 配置，请检查 JSON 文件及字段。") from exc
    base = path.parent
    model = settings.model
    checkpoint = require_checkpoint(resolve_path(base, model.checkpoint_path))
    if model.kind == "python":
        try:
            backend = load_symbol(model.factory)(checkpoint_path=str(checkpoint))
            classifier = PythonModelAdapter(backend)
        except AgentError:
            raise
        except Exception as exc:
            raise AgentError("MODEL_NOT_READY", "加载已有模型失败，请检查权重与模型工厂。") from exc
    else:
        classifier = ScriptModelAdapter(
            [sys.executable if arg == "{python}" else arg for arg in model.command],
            checkpoint,
            cwd=resolve_path(base, model.cwd),
            timeout=model.timeout_seconds,
        )
    preprocessor = DefaultTextPreprocessor()
    if settings.preprocessor_factory:
        try:
            preprocessor = load_symbol(settings.preprocessor_factory)()
        except AgentError:
            raise
        except Exception as exc:
            raise AgentError("CONFIG_ERROR", "无法创建文本预处理器。") from exc
    return ClassificationAgent(
        classifier,
        CategoryCatalog.from_files(
            resolve_path(base, settings.labels_path), resolve_path(base, settings.categories_path)
        ),
        preprocessor=preprocessor,
        config=settings.agent,
        trace_sink=JsonlTraceSink(resolve_path(base, settings.trace_path))
        if settings.trace_path
        else None,
    )


def create_demo_agent(
    *, config: AgentConfig | None = None, trace_path: str | Path | None = None
) -> ClassificationAgent:
    """Explicit opt-in: five synthetic labels and keyword scores, no weights or training."""
    data = files("agent").joinpath("data")
    if config is None:
        config = AgentConfig(
            confidence=ConfidencePolicy(
                model_version="demo-keywords-v1", min_probability=0.6, min_margin=0.15
            )
        )
    return ClassificationAgent(
        PythonModelAdapter(DemoClassifier()),
        CategoryCatalog.from_files(
            data.joinpath("demo_labels.json"), data.joinpath("demo_categories.json")
        ),
        config=config,
        trace_sink=JsonlTraceSink(trace_path) if trace_path else None,
    )


def create_medbert_agent(
    *,
    checkpoint_path: str | Path | None = None,
    device: str | None = None,
    top_k: int = 5,
    confidence_threshold: float = 0.6,
    trace_path: str | Path | None = None,
) -> ClassificationAgent:
    """Load local weights with member 2 preprocessing; never train at startup."""
    from agent.integrations.medbert_backend import (
        DEFAULT_CHECKPOINT,
        MedBertBackend,
        catalog_from_bundle,
    )
    from agent.integrations.team_preprocessor import TeamPreprocessor

    backend = MedBertBackend(checkpoint_path or DEFAULT_CHECKPOINT, device=device)
    return ClassificationAgent(
        PythonModelAdapter(backend),
        catalog_from_bundle(backend.bundle),
        preprocessor=TeamPreprocessor(),
        config=AgentConfig(
            top_k=top_k,
            confidence=ConfidencePolicy(
                model_version=backend.describe().model_version,
                min_probability=confidence_threshold,
                min_margin=0,
            ),
        ),
        trace_sink=JsonlTraceSink(trace_path) if trace_path else None,
    )
