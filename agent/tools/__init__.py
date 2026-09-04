"""Independently callable workflow tools."""

from agent.tools.base import BaseTool, ToolResult, ToolStep
from agent.tools.classify_text import ClassifyTextTool
from agent.tools.clean_dataset import CleanDatasetTool
from agent.tools.collect_data import CollectDataTool
from agent.tools.query_category import QueryCategoryTool
from agent.tools.serve_web import ServeWebTool
from agent.tools.train_model import TrainModelTool

TOOL_REGISTRY = {
    "collect_data": CollectDataTool,
    "clean_dataset": CleanDatasetTool,
    "train_model": TrainModelTool,
    "classify_text": ClassifyTextTool,
    "query_category": QueryCategoryTool,
    "serve_web": ServeWebTool,
}


def get_tool(name: str) -> BaseTool:
    if name not in TOOL_REGISTRY:
        raise ValueError(f"未注册的工具：{name}")
    return TOOL_REGISTRY[name]()


def get_all_tool_specs() -> list:
    return [cls().to_llm_spec() for cls in TOOL_REGISTRY.values()]


__all__ = [
    "BaseTool",
    "ToolResult",
    "ToolStep",
    "TOOL_REGISTRY",
    "get_tool",
    "get_all_tool_specs",
    "CollectDataTool",
    "CleanDatasetTool",
    "TrainModelTool",
    "ClassifyTextTool",
    "QueryCategoryTool",
    "ServeWebTool",
]
