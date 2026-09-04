"""Member 5 facade: no FastAPI dependency or shared request state."""

import logging
import re
from threading import Lock

from agent.errors import AgentError
from agent.integrations.team_contract import failure_result, to_tool_result, to_tool_step
from agent.tools.base import ToolResult

logger = logging.getLogger(__name__)


class AcademicClassificationAgent:
    def __init__(
        self, llm_client=None, *, core=None, checkpoint_path=None, device=None, trace_path=None
    ):
        self._core = core
        self._load_lock = Lock()
        self._settings = {
            "checkpoint_path": checkpoint_path,
            "device": device,
            "trace_path": trace_path,
        }
        # Constructor compatibility; routing uses the explicit LangGraph workflow.
        self.llm_client = llm_client

    def _get_core(self):
        if self._core is None:
            with self._load_lock:
                if self._core is None:
                    from agent.factory import create_medbert_agent
                    from config import AGENT_CONFIG

                    self._core = create_medbert_agent(
                        **self._settings,
                        top_k=AGENT_CONFIG.get("top_k", 5),
                        confidence_threshold=AGENT_CONFIG.get("confidence_threshold", 0.6),
                    )
        return self._core

    @property
    def tool_specs(self):
        from agent.tools import get_all_tool_specs

        return get_all_tool_specs()

    def warmup(self) -> dict:
        """Optional backend startup hook; raises AgentError if the model is unavailable."""
        core = self._get_core()
        backend = getattr(core.classifier, "backend", None)
        bundle = getattr(backend, "bundle", {})
        return {
            "model_info": core.model_info.model_dump(mode="json"),
            "deployment": {key: value for key, value in bundle.items() if key != "labels"},
        }

    def _classify(self, method, *args, on_step=None, **kwargs) -> ToolResult:
        try:
            core = self._get_core()
            callback = (lambda step: on_step(to_tool_step(step).to_dict())) if on_step else None
            result = getattr(core, method)(*args, on_step=callback, **kwargs)
            backend = getattr(core.classifier, "backend", None)
            return to_tool_result(result, deployment=getattr(backend, "bundle", None))
        except Exception as exc:
            logger.exception("Classification service failed")
            return failure_result(exc)

    def classify_text(self, text, *, top_k=None, confidence_threshold=None, on_step=None):
        return self._classify(
            "classify_text",
            text,
            top_k=top_k,
            confidence_threshold=confidence_threshold,
            on_step=on_step,
        )

    def classify_file(
        self, content: bytes, filename: str, *, top_k=None, confidence_threshold=None, on_step=None
    ):
        return self._classify(
            "classify_file",
            content,
            filename,
            top_k=top_k,
            confidence_threshold=confidence_threshold,
            on_step=on_step,
        )

    def classify_paper(
        self,
        *,
        title="",
        keywords=None,
        abstract="",
        top_k=None,
        confidence_threshold=None,
        on_step=None,
    ):
        from preprocess.clean import parse_keywords

        if (keywords is not None and not isinstance(keywords, (list, str))) or (
            isinstance(keywords, list) and any(not isinstance(item, str) for item in keywords)
        ):
            return failure_result(AgentError("INVALID_INPUT", "关键词必须是字符串或字符串列表。"))
        return self._classify(
            "classify_paper",
            title=title,
            keywords=parse_keywords(keywords),
            abstract=abstract,
            top_k=top_k,
            confidence_threshold=confidence_threshold,
            on_step=on_step,
        )

    def plan(self, user_input: str) -> list[dict]:
        if not isinstance(user_input, str) or not user_input.strip():
            return []
        text = user_input.strip()
        # Exact commands only: an article mentioning “训练” remains classification input.
        if text in {"训练", "训练模型"}:
            return [{"tool": "train_model", "args": {"model_type": "medbert"}}]
        if text in {"清洗", "清洗数据", "清洗数据集"}:
            return [{"tool": "clean_dataset", "args": {}}]
        if text in {"采集", "采集数据", "爬取数据"}:
            return [{"tool": "collect_data", "args": {"categories": []}}]
        text = re.sub(r"^(?:分类|预测)\s*[:：]\s*", "", text, count=1)
        return [{"tool": "classify_text", "args": {"text": text}}]

    def execute_step(self, tool_name: str, args: dict) -> ToolResult:
        try:
            if tool_name == "classify_text":
                return self.classify_text(**args)
            if tool_name == "classify_paper":
                return self.classify_paper(**args)
            from agent.tools import get_tool

            return get_tool(tool_name).run(**args)
        except Exception as exc:
            return failure_result(exc)

    def run(self, user_input: str | dict) -> ToolResult:
        """Plain text, exact offline commands, or {task: tool_name, ...arguments}."""
        if isinstance(user_input, dict):
            payload = dict(user_input)
            task = payload.pop("task", None)
            if not isinstance(task, str) or task not in {
                "classify_text",
                "classify_paper",
                "collect_data",
                "clean_dataset",
                "train_model",
                "query_category",
            }:
                return failure_result(AgentError("INVALID_INPUT", "未知任务类型。"))
            return self.execute_step(task, payload)
        plan = self.plan(user_input)
        if not plan:
            return failure_result(AgentError("EMPTY_TEXT", "请输入分类文本或明确的任务参数。"))
        return self.execute_step(plan[0]["tool"], plan[0]["args"])


_default_agent = None
_default_lock = Lock()


def get_agent() -> AcademicClassificationAgent:
    global _default_agent
    with _default_lock:
        if _default_agent is None:
            _default_agent = AcademicClassificationAgent()
        return _default_agent
