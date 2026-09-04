"""Compatibility tool: classification decisions are owned by the workflow."""

from agent.tools.base import BaseTool, ToolResult


class ClassifyTextTool(BaseTool):
    name = "classify_text"
    description = "对文本执行预处理、MedBERT 推理、置信度判断和类别查询。"
    input_schema = {
        "type": "object",
        "required": ["text"],
        "properties": {
            "text": {"type": "string"},
            "top_k": {"type": "integer", "minimum": 1, "default": 5},
            "confidence_threshold": {"type": "number", "minimum": 0, "maximum": 1, "default": 0.6},
        },
    }

    def run(
        self, text=None, file_path=None, top_k=5, confidence_threshold=0.6, **kwargs
    ) -> ToolResult:
        if file_path and not text:
            return ToolResult.fail(
                "上传文件请通过 get_agent().classify_file(content, filename) 处理。"
            )
        from agent import get_agent

        return get_agent().classify_text(
            text,
            top_k=top_k,
            confidence_threshold=confidence_threshold,
        )
