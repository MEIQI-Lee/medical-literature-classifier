"""
Tool 基类与统一返回结构
================================
所有成员(成员1~5)各自实现的Tool必须继承 BaseTool，并遵循统一接口契约。

设计原则:
1. 统一输入(**kwargs) + 统一输出(ToolResult)
2. 步骤(steps)与日志(logs)向上暴露，供Agent前端过程展示
3. 幂等可重入: 相同输入应能从断点续跑
4. 异常不吞没: 失败时返回 ToolResult.fail()，由LLM决策重试/跳过
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from typing import Any, Optional, List
import time
import uuid


@dataclass
class ToolStep:
    """Agent执行步骤(供前端过程展示)"""
    step_id: str
    name: str
    status: str = "pending"  # pending / running / done / failed
    detail: str = ""
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ToolResult:
    """Tool统一返回结构"""
    success: bool
    data: dict = field(default_factory=dict)
    error: Optional[str] = None
    steps: List[ToolStep] = field(default_factory=list)
    logs: List[str] = field(default_factory=list)

    @classmethod
    def ok(cls, data: dict, steps: list = None, logs: list = None) -> "ToolResult":
        return cls(success=True, data=data,
                   steps=steps or [], logs=logs or [])

    @classmethod
    def fail(cls, error: str, steps: list = None, logs: list = None) -> "ToolResult":
        return cls(success=False, error=error,
                   steps=steps or [], logs=logs or [])

    def to_dict(self) -> dict:
        return {
            "success": self.success,
            "data": self.data,
            "error": self.error,
            "steps": [s.to_dict() for s in self.steps],
            "logs": self.logs,
        }


class BaseTool(ABC):
    """
    所有Tool的基类
    ================================
    子类必须实现:
        name: str          - Tool唯一标识(如 "collect_data")
        description: str   - 给LLM的功能描述
        input_schema: dict - 输入参数schema(JSON Schema格式)
        run(**kwargs)      - 执行逻辑
    """

    name: str = ""
    description: str = ""
    input_schema: dict = {}

    def __init__(self):
        self.steps: List[ToolStep] = []
        self.logs: List[str] = []

    # ---------- 工具方法(供子类调用) ----------
    def add_step(self, name: str, detail: str = "", status: str = "running") -> ToolStep:
        """新增执行步骤并返回,后续可更新状态"""
        step = ToolStep(step_id=uuid.uuid4().hex[:8],
                        name=name, status=status, detail=detail)
        self.steps.append(step)
        return step

    def update_step(self, step: ToolStep, status: str, detail: str = ""):
        """更新步骤状态"""
        step.status = status
        if detail:
            step.detail = detail

    def log(self, msg: str):
        """追加日志"""
        self.logs.append(f"[{time.strftime('%H:%M:%S')}] {msg}")

    # ---------- LLM可读描述 ----------
    def to_llm_spec(self) -> dict:
        """转换为LLM可读的Tool描述(供function calling)"""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
        }

    # ---------- 子类必须实现 ----------
    @abstractmethod
    def run(self, **kwargs) -> ToolResult:
        """执行Tool,返回统一ToolResult"""
        raise NotImplementedError
