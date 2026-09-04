import logging
import math
from collections.abc import Callable
from functools import partial
from time import perf_counter
from uuid import uuid4

from pydantic import ValidationError

from agent.adapters import Classifier
from agent.catalog import CategoryCatalog
from agent.errors import AgentError
from agent.graph import InputRequest, WorkflowState, build_graph
from agent.schemas import (
    AgentConfig,
    AgentResult,
    ConfidencePolicy,
    ErrorInfo,
    InputSummary,
    ModelInfo,
    PaperFields,
    ParsedDocument,
    PreparedText,
    StepInfo,
    WarningInfo,
)
from agent.tools.documents import parse_document
from agent.tools.prediction import classify_text, query_categories, select_candidates
from agent.tools.text import DefaultTextPreprocessor, Preprocessor
from agent.tracing import JsonlTraceSink

logger = logging.getLogger(__name__)

# Preserve the upstream import path as well as `from agent import get_agent`.
from agent.service import AcademicClassificationAgent, get_agent  # noqa: E402,F401


class ClassificationAgent:
    def __init__(
        self,
        classifier: Classifier,
        catalog: CategoryCatalog,
        *,
        preprocessor: Preprocessor | None = None,
        config: AgentConfig | None = None,
        trace_sink: JsonlTraceSink | None = None,
    ):
        self.classifier = classifier
        self.catalog = catalog
        self.preprocessor = preprocessor or DefaultTextPreprocessor()
        self.config = (config or AgentConfig()).model_copy(deep=True)
        self.trace_sink = trace_sink
        self.model_info = ModelInfo.model_validate(classifier.describe()).model_copy(deep=True)
        catalog.validate_model(self.model_info)
        if not isinstance(getattr(self.preprocessor, "version", None), str) or not callable(
            getattr(self.preprocessor, "prepare", None)
        ):
            raise AgentError("CONFIG_ERROR", "预处理器必须提供 version 字符串和 prepare 方法。")
        if self.preprocessor.version != self.model_info.preprocess_version:
            raise AgentError("PREPROCESS_VERSION_MISMATCH", "模型与文本预处理规则的版本不匹配。")
        policy = self.config.confidence
        if policy and (
            policy.model_version != self.model_info.model_version
            or self.model_info.score_type != "probability"
        ):
            raise AgentError("CONFIG_ERROR", "置信度阈值必须对应当前模型及其概率输出。")

        tools = {
            "validate_input": self._validate_input,
            "read_text": self._read_text,
            "parse_document": self._parse_document,
            "prepare_text": self._prepare_text,
            "classify_text": self._classify_text,
            "select_candidates": self._select_candidates,
            "accept_prediction": self._accept_prediction,
            "review_candidates": self._review_candidates,
            "query_category": self._query_category,
            "generate_result": self._generate_result,
        }
        self.graph = build_graph(
            {
                name: partial(self._run_step, name=name, operation=operation)
                for name, operation in tools.items()
            }
        )

    def classify_text(
        self,
        text: str,
        *,
        top_k: int | None = None,
        confidence_threshold: float | None = None,
        on_step: Callable[[StepInfo], None] | None = None,
    ) -> AgentResult:
        return self._run(
            InputRequest("text", text, top_k=top_k, confidence_threshold=confidence_threshold),
            on_step,
        )

    def classify_file(
        self,
        content: bytes,
        filename: str,
        *,
        top_k: int | None = None,
        confidence_threshold: float | None = None,
        on_step: Callable[[StepInfo], None] | None = None,
    ) -> AgentResult:
        return self._run(
            InputRequest("file", content, filename, top_k, confidence_threshold), on_step
        )

    def classify_paper(
        self,
        *,
        title: str = "",
        abstract: str = "",
        keywords: list[str] | None = None,
        top_k: int | None = None,
        confidence_threshold: float | None = None,
        on_step: Callable[[StepInfo], None] | None = None,
    ) -> AgentResult:
        return self._run(
            InputRequest(
                "paper",
                {
                    "title": title,
                    "abstract": abstract,
                    "keywords": [] if keywords is None else keywords,
                },
                top_k=top_k,
                confidence_threshold=confidence_threshold,
            ),
            on_step,
        )

    @staticmethod
    def _emit(callback, step, notes):
        if callback is not None:
            try:
                callback(step.model_copy(deep=True))
            except Exception:
                if not any(item.code == "STEP_CALLBACK_FAILED" for item in notes):
                    notes.append(
                        WarningInfo(
                            code="STEP_CALLBACK_FAILED",
                            message="部分进度通知发送失败，分类流程继续执行。",
                        )
                    )

    def _run_step(self, state: WorkflowState, *, name: str, operation: Callable) -> dict:
        started = perf_counter()
        notes = list(state["warnings"])
        sequence = len(state["steps"]) + 1
        event = StepInfo(
            request_id=state["request_id"],
            sequence=sequence,
            step=name,
            status="running",
            message="开始处理",
        )
        self._emit(state.get("on_step"), event, notes)
        try:
            updates, message = operation(state)
            notes.extend(updates.pop("warnings", []))
            status = "success"
        except Exception as exc:
            if isinstance(exc, AgentError):
                code, message = exc.code, exc.message
            else:
                logger.error("Workflow step %s raised %s", name, type(exc).__name__)
                code, message = "INTERNAL_ERROR", "处理过程中出现异常，请联系系统维护者。"
            updates = {"error": ErrorInfo(code=code, message=message, step=name)}
            status = "error"
        terminal = event.model_copy(
            update={
                "status": status,
                "duration_ms": round((perf_counter() - started) * 1000, 3),
                "message": message,
            }
        )
        self._emit(state.get("on_step"), terminal, notes)
        return {**updates, "steps": [*state["steps"], terminal], "warnings": notes}

    def _run(self, request: InputRequest, on_step) -> AgentResult:
        started = perf_counter()
        notes = []
        if self.model_info.mode == "demo":
            notes.append(
                WarningInfo(
                    code="DEMO_MODE",
                    message="当前使用模拟分类器及演示类别，结果不代表真实模型效果或 CLC 分类。",
                )
            )
        initial: WorkflowState = {
            "request_id": str(uuid4()),
            "request": request,
            "on_step": on_step,
            "steps": [],
            "warnings": notes,
            "error": None,
        }
        state = self.graph.invoke(initial)
        error = state.get("error")
        document = state.get("document")
        prepared = state.get("prepared")
        output = state.get("output")
        candidates = state.get("candidates", []) if not error else []
        policy = state.get("confidence_policy")
        result = AgentResult(
            request_id=state["request_id"],
            status="error" if error else "success",
            mode=self.model_info.mode,
            prediction=candidates[0] if candidates else None,
            candidates=candidates,
            confidence_status=state.get("confidence_status", "unknown") if not error else "unknown",
            confidence_threshold=policy.min_probability if policy else None,
            requested_top_k=state.get("top_k", self.config.top_k),
            returned_top_k=len(candidates),
            display_mode="error" if error else state["display_mode"],
            message=state["message"],
            model_info=self.model_info.model_copy(deep=True),
            input=InputSummary(
                source_type=document.source_type if document else request.kind,
                filename=document.filename if document else None,
                pages=document.pages if document else None,
                extracted_chars=len(document.text) if document else 0,
                prepared_chars=len(prepared.text) if prepared else 0,
                preparation_strategy=prepared.strategy if prepared else None,
                truncated=output.truncated if output else False,
                input_tokens=output.input_tokens if output else None,
            ),
            steps=state["steps"],
            warnings=state["warnings"],
            error=error,
            duration_ms=round((perf_counter() - started) * 1000, 3),
        )
        if self.trace_sink is not None:
            try:
                self.trace_sink.write(result)
            except OSError:
                result.warnings.append(
                    WarningInfo(
                        code="TRACE_WRITE_FAILED", message="执行日志保存失败，分类结果仍可使用。"
                    )
                )
        return result

    def _validate_input(self, state):
        request = state["request"]
        top_k = self.config.top_k if request.top_k is None else request.top_k
        if type(top_k) is not int or top_k < 1:
            raise AgentError("INVALID_INPUT", "top_k 必须是大于 0 的整数。")
        policy = self.config.confidence
        threshold = request.confidence_threshold
        if threshold is not None:
            if (
                type(threshold) not in (int, float)
                or not 0 <= threshold <= 1
                or not math.isfinite(threshold)
                or self.model_info.score_type != "probability"
            ):
                raise AgentError("INVALID_INPUT", "概率阈值必须是 0 到 1 之间的有限数值。")
            policy = ConfidencePolicy(
                model_version=self.model_info.model_version,
                min_probability=threshold,
                min_margin=policy.min_margin if policy else 0,
            )
        if request.kind == "file":
            if not isinstance(request.filename, str) or not request.filename.strip():
                raise AgentError("INVALID_INPUT", "请提供文件名。")
            if not isinstance(request.payload, bytes):
                raise AgentError("INVALID_INPUT", "文件内容必须为 bytes。")
            if not request.payload:
                raise AgentError("EMPTY_FILE", "上传文件为空。")
            if len(request.payload) > self.config.max_file_bytes:
                raise AgentError("FILE_TOO_LARGE", "上传文件超过大小限制。")
        else:
            if request.kind == "paper":
                try:
                    fields = PaperFields.model_validate(request.payload)
                except ValidationError as exc:
                    raise AgentError(
                        "INVALID_INPUT", "标题、摘要必须是字符串，关键词必须是字符串列表。"
                    ) from exc
                raw = "\n".join([fields.title, *fields.keywords, fields.abstract])
            else:
                raw = request.payload
            if not isinstance(raw, str):
                raise AgentError("INVALID_INPUT", "输入文本必须是字符串。")
            if not raw.strip():
                raise AgentError("EMPTY_TEXT", "请输入需要分类的文本。")
            if len(raw) > self.config.max_text_chars:
                raise AgentError("TEXT_TOO_LONG", "输入文本超过处理上限，请缩短文本。")
        return {"top_k": top_k, "confidence_policy": policy}, "输入校验完成"

    def _read_text(self, state):
        request = state["request"]
        if request.kind == "paper":
            fields = PaperFields.model_validate(request.payload)
            document = ParsedDocument(
                text="\n".join([fields.title, *fields.keywords, fields.abstract]),
                fields=fields,
                source_type="paper",
            )
        else:
            document = ParsedDocument(text=request.payload, source_type="text")
        return {"document": document}, f"读取文本完成，共 {len(document.text)} 字符"

    def _parse_document(self, state):
        request = state["request"]
        document = parse_document(request.payload, request.filename, self.config)
        return {"document": document, "warnings": document.warnings}, (
            f"文档解析完成，共 {len(document.text)} 字符"
        )

    def _prepare_text(self, state):
        prepared = PreparedText.model_validate(self.preprocessor.prepare(state["document"]))
        if not prepared.text.strip():
            raise AgentError("EMPTY_TEXT", "清洗后没有有效文本，请补充正文或摘要。")
        if len(prepared.text) < self.config.min_text_chars:
            raise AgentError("TEXT_TOO_SHORT", "有效文本过短，请补充标题、摘要或关键词。")
        if len(prepared.text) > self.config.max_text_chars:
            raise AgentError("TEXT_TOO_LONG", "整理后的文本超过处理上限，请缩短文本。")
        return {"prepared": prepared, "warnings": prepared.warnings}, (
            f"文本整理完成，有效字符数为 {len(prepared.text)}"
        )

    def _classify_text(self, state):
        output = classify_text(
            self.classifier,
            state["prepared"].text,
            top_k=min(max(3, state["top_k"]), len(self.model_info.label_ids)),
            expected_info=self.model_info,
        )
        notes = []
        if output.truncated:
            notes.append(
                WarningInfo(
                    code="MODEL_INPUT_TRUNCATED",
                    message="文本超过模型输入长度，预测使用了截断后的内容。",
                )
            )
        if output.model_info.score_type == "probability" and not output.model_info.calibrated:
            notes.append(
                WarningInfo(
                    code="UNCALIBRATED_PROBABILITY",
                    message="模型概率尚未校准，不能视为实际正确率。",
                )
            )
        return {"output": output, "warnings": notes}, "模型推理完成"

    def _select_candidates(self, state):
        confidence = select_candidates(state["output"], state["confidence_policy"])
        notes = []
        if confidence == "unknown":
            notes.append(
                WarningInfo(
                    code="CONFIDENCE_UNAVAILABLE",
                    message="当前输出未启用可靠性阈值判断，请结合候选类别核对。",
                )
            )
        return {"confidence_status": confidence, "warnings": notes}, "候选排序与置信度判断完成"

    def _accept_prediction(self, state):
        return {"display_mode": "top1"}, "突出显示首选类别，同时保留候选列表"

    def _review_candidates(self, state):
        return {"display_mode": "candidates"}, "展示多个候选类别供核对"

    def _query_category(self, state):
        candidates = query_categories(state["output"], self.catalog)
        count = state["top_k"]
        if state["confidence_status"] != "normal":
            count = max(3, count)
        candidates = candidates[:count]
        return {"candidates": candidates}, f"已补全 {len(candidates)} 个类别的编码与名称"

    def _generate_result(self, state):
        if state.get("error"):
            message = state["error"].message
        elif state["confidence_status"] == "normal":
            message = "分类完成。"
        elif state["confidence_status"] == "low":
            message = "当前文本的分类结果不够明确，请参考候选类别或补充摘要和关键词。"
        else:
            message = "已生成候选类别，建议核对分类结果。"
        if self.model_info.mode == "demo":
            message = "【演示模式】" + message
        return {"message": message}, "结果整理完成"
