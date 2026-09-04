"""Bridge the existing crawler to the team's ToolResult protocol."""

import json
import math

from agent.tools.base import BaseTool, ToolResult


class CollectDataTool(BaseTool):
    name = "collect_data"
    description = "按指定 CLC 类别调用已有爬虫；目标总量按类别平均换算为采集上限。"
    input_schema = {
        "type": "object",
        "required": ["categories"],
        "properties": {
            "categories": {"type": "array", "items": {"type": "string"}, "minItems": 1},
            "target_count": {"type": "integer", "minimum": 1, "default": 3000},
            "resume": {"type": "boolean", "default": True},
            "headless": {"type": "boolean", "default": False},
        },
    }

    def run(self, categories=None, target_count=3000, resume=True, headless=False, **kwargs):
        self.steps, self.logs = [], []
        try:
            if (
                not isinstance(categories, list)
                or not categories
                or any(not isinstance(code, str) for code in categories)
            ):
                raise ValueError("请明确提供要采集的类别列表。")
            if type(target_count) is not int or target_count < 1:
                raise ValueError("target_count 必须是正整数。")
            if type(resume) is not bool or type(headless) is not bool:
                raise ValueError("resume 和 headless 必须是布尔值。")
            from config import CATEGORY_MAPPING_FILE

            mapping = json.loads(CATEGORY_MAPPING_FILE.read_text(encoding="utf-8"))
            categories = list(dict.fromkeys(categories))
            if not set(categories).issubset(mapping):
                raise ValueError("采集列表包含未知类别。")
            per_category = math.ceil(target_count / len(categories))
            from crawler.cnki import run_full_crawl

            step = self.add_step("collect_data", f"每类最多 {per_category} 条")
            report = run_full_crawl(
                categories=categories,
                max_per_category=per_category,
                resume=resume,
                headless=headless,
                save_per_category=True,
                fetch_detail=True,
            )
            failed = report.get("failed_categories", [])
            self.update_step(step, "failed" if failed else "done", "采集完成，请核对实际条数。")
            return ToolResult(
                success=not failed,
                data={
                    "collection": report,
                    "requested_target_count": target_count,
                    "max_per_category": per_category,
                    "effective_target_limit": per_category * len(categories),
                },
                error="部分类别采集失败，请查看 collection.failed_categories。" if failed else None,
                steps=self.steps,
                logs=self.logs,
            )
        except Exception as exc:
            if self.steps:
                self.update_step(self.steps[-1], "failed", str(exc))
            return ToolResult.fail(str(exc), steps=self.steps, logs=self.logs)
