"""Stable member 5 contract, compatible with the upstream ToolResult envelope."""

from uuid import uuid4

from agent.errors import AgentError
from agent.schemas import AgentResult, StepInfo
from agent.tools.base import ToolResult, ToolStep


def to_tool_step(step: StepInfo) -> ToolStep:
    return ToolStep(
        step_id=f"{step.request_id}:{step.sequence}",
        name=step.step,
        status={"running": "running", "success": "done", "error": "failed"}[step.status],
        detail=step.message,
    )


def to_tool_result(result: AgentResult, *, deployment: dict | None = None) -> ToolResult:
    deployment = deployment or {}
    topk = [
        {"code": item.category_code, "name": item.category_name, "confidence": item.confidence}
        for item in result.candidates
    ]
    data = {
        "request_id": result.request_id,
        "prediction": {
            "top1": topk[0] if topk else None,
            "topk": topk,
            "model_type": deployment.get("model_type", result.model_info.model_version),
            "checkpoint": deployment.get("checkpoint"),
        },
        "low_confidence": result.confidence_status == "low",
        "confidence_status": result.confidence_status,
        "confidence_threshold": result.confidence_threshold,
        "display_mode": result.display_mode,
        "message": result.message,
        "requested_top_k": result.requested_top_k,
        "returned_top_k": result.returned_top_k,
        "model_info": result.model_info.model_dump(mode="json"),
        "input": result.input.model_dump(mode="json"),
        "warnings": [warning.model_dump() for warning in result.warnings],
        "duration_ms": result.duration_ms,
        "error_code": result.error.code if result.error else None,
    }
    return ToolResult(
        success=result.status == "success",
        data=data,
        error=result.error.message if result.error else None,
        steps=[to_tool_step(step) for step in result.steps],
        logs=[step.message for step in result.steps],
    )


def failure_result(exc: Exception) -> ToolResult:
    code = exc.code if isinstance(exc, AgentError) else "INTERNAL_ERROR"
    message = exc.message if isinstance(exc, AgentError) else "处理失败，请检查服务日志。"
    return ToolResult(
        success=False,
        data={"request_id": str(uuid4()), "error_code": code},
        error=message,
        steps=[ToolStep(step_id=uuid4().hex, name="initialize", status="failed", detail=message)],
    )
