"""Read category names without loading model weights."""

import json

from agent.tools.base import BaseTool, ToolResult


class QueryCategoryTool(BaseTool):
    name = "query_category"
    description = "查询类别名称及当前 MedBERT 是否支持该类别。"
    input_schema = {
        "type": "object",
        "required": ["code"],
        "properties": {"code": {"type": "string"}},
    }

    def run(self, code=None, **kwargs):
        from config import CATEGORY_MAPPING_FILE, CHECKPOINT_DIR

        try:
            mapping = json.loads(CATEGORY_MAPPING_FILE.read_text(encoding="utf-8"))
            labels = json.loads(
                (CHECKPOINT_DIR / "medbert_v20/best/labels.json").read_text(encoding="utf-8")
            )
            mapping.update(labels["code2name"])
            if not isinstance(code, str) or code not in mapping:
                return ToolResult.fail("未找到该类别编码。")
            return ToolResult.ok(
                {
                    "code": code,
                    "name": mapping[code],
                    "supported": code in labels["label2id"],
                }
            )
        except (OSError, ValueError, KeyError):
            return ToolResult.fail("无法读取类别映射。")
