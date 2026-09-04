"""
Tool 5: Web可视化服务 (成员5负责)
================================
职责: 启动前后端服务,提供分类API与可视化页面
注: 此Tool更多作为后端入口,实际由成员5独立维护 frontend/ + backend/
"""
from agent.tools.base import BaseTool, ToolResult


class ServeWebTool(BaseTool):
    """Web可视化服务启动工具"""

    name = "serve_web"
    description = (
        "启动Web可视化分类网站后端服务(FastAPI),提供: "
        "文本分类API、文件上传分类API、数据统计API、模型性能API、"
        "Agent执行过程展示API。前端(Vue3)独立部署于 frontend/。"
    )
    input_schema = {
        "type": "object",
        "properties": {
            "host": {"type": "string", "default": "0.0.0.0"},
            "port": {"type": "integer", "default": 8000},
            "reload": {"type": "boolean", "default": False}
        }
    }

    def run(self, host: str = "0.0.0.0", port: int = 8000,
            reload: bool = False, **kwargs) -> ToolResult:
        # TODO(成员5): 实际由 backend/main.py 提供,此Tool仅作Agent编排占位
        # import uvicorn
        # uvicorn.run("backend.main:app", host=host, port=port, reload=reload)
        step = self.add_step("serve_web", "服务待实现", "pending")
        return ToolResult.fail(
            error="serve_web 尚未实现,待成员5填充 backend/main.py",
            steps=self.steps, logs=self.logs
        )
