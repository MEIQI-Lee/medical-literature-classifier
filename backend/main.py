"""HTTP API and static web entry point for the academic classification demo."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from agent import get_agent
from agent.errors import AgentError
from config import RESULTS_DIR, WEB_CONFIG

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
SUPPORTED_SUFFIXES = {".txt", ".pdf", ".docx"}
INPUT_ERROR_CODES = {
    "EMPTY_TEXT", "TEXT_TOO_SHORT", "INVALID_INPUT", "UNSUPPORTED_FILE_TYPE", "NO_EXTRACTABLE_TEXT"
}

app = FastAPI(
    title="学术文本分类智能体系统",
    version="1.0.0",
    description="面向《中国图书馆分类法》R类医学文献的文本分类服务。",
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class ClassifyRequest(BaseModel):
    text: str = Field(..., min_length=1, description="待分类的医学文本")
    top_k: int = Field(default=5, ge=1, le=110, description="返回候选类别数")
    confidence_threshold: float | None = Field(default=None, ge=0, le=1)


class AgentRequest(BaseModel):
    user_input: str | dict[str, Any]


def _load_json(path: Path) -> dict[str, Any]:
    """Load a checked-in report without exposing a server-side traceback."""
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"未找到报告文件：{path.name}") from exc
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=500, detail=f"报告文件格式错误：{path.name}") from exc
    if not isinstance(payload, dict):
        raise HTTPException(status_code=500, detail=f"报告文件内容无效：{path.name}")
    return payload


def _http_status(payload: Mapping[str, Any]) -> int:
    if payload.get("success"):
        return 200
    code = str(payload.get("data", {}).get("error_code") or "")
    if code == "FILE_TOO_LARGE":
        return 413
    if code == "MODEL_NOT_READY":
        return 503
    if code in INPUT_ERROR_CODES:
        return 422
    return 500


def _result_response(result: Any) -> JSONResponse:
    payload = result.to_dict()
    return JSONResponse(status_code=_http_status(payload), content=payload)


def _sse(event: str, payload: Any) -> str:
    """Encode one server-sent event without escaping Chinese workflow details."""
    return f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"


async def _stream_classification(invoke: Any):
    """Run the synchronous agent in a thread and relay each callback immediately."""
    loop = asyncio.get_running_loop()
    events: asyncio.Queue[tuple[str, dict[str, Any]]] = asyncio.Queue()

    def emit(event: str, payload: dict[str, Any]) -> None:
        loop.call_soon_threadsafe(events.put_nowait, (event, payload))

    def on_step(step: dict[str, Any]) -> None:
        emit("step", step)

    def run() -> None:
        try:
            emit("result", invoke(on_step).to_dict())
        except Exception:
            emit(
                "result",
                {
                    "success": False,
                    "data": {"error_code": "MODEL_INFERENCE_FAILED"},
                    "error": "分类服务执行失败，请检查模型配置后重试。",
                    "steps": [],
                    "logs": [],
                },
            )

    worker = asyncio.create_task(asyncio.to_thread(run))
    yield _sse("status", {"step_id": "server:prepare", "name": "load_model", "status": "running", "detail": "正在准备分类模型"})
    while True:
        event, payload = await events.get()
        yield _sse(event, payload)
        if event == "result":
            break
    await worker


def _stream_response(invoke: Any) -> StreamingResponse:
    return StreamingResponse(
        _stream_classification(invoke),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/api/health")
async def health(check_model: bool = Query(default=False)) -> dict[str, Any]:
    """Report process readiness; loading the model remains opt-in and lazy."""
    response: dict[str, Any] = {"status": "ok", "model_checked": False}
    if not check_model:
        return response
    try:
        metadata = await asyncio.to_thread(get_agent().warmup)
    except AgentError as exc:
        return {"status": "degraded", "model_checked": True, "error_code": exc.code, "detail": exc.message}
    except Exception:
        return {"status": "degraded", "model_checked": True, "error_code": "MODEL_NOT_READY"}
    response.update({"model_checked": True, "model": metadata.get("model_info", {})})
    return response


@app.post("/api/classify")
async def classify_text(req: ClassifyRequest) -> JSONResponse:
    """Classify directly supplied text through the member-4 workflow."""
    result = await asyncio.to_thread(
        get_agent().classify_text,
        req.text,
        top_k=req.top_k,
        confidence_threshold=req.confidence_threshold,
    )
    return _result_response(result)


@app.post("/api/classify/stream")
async def classify_text_stream(req: ClassifyRequest) -> StreamingResponse:
    """Classify text while publishing each agent workflow event to the browser."""
    return _stream_response(
        lambda on_step: get_agent().classify_text(
            req.text,
            top_k=req.top_k,
            confidence_threshold=req.confidence_threshold,
            on_step=on_step,
        )
    )


@app.post("/api/classify/file")
async def classify_file(
    file: UploadFile = File(...),
    top_k: int = Query(default=5, ge=1, le=110),
    confidence_threshold: float | None = Query(default=None, ge=0, le=1),
) -> JSONResponse:
    """Classify TXT, PDF, or DOCX content without persisting user uploads."""
    filename = file.filename or ""
    if Path(filename).suffix.lower() not in SUPPORTED_SUFFIXES:
        raise HTTPException(status_code=422, detail="仅支持 TXT、PDF 和 DOCX 文件。")
    try:
        content = await file.read(MAX_UPLOAD_BYTES + 1)
    finally:
        await file.close()
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="上传文件不能超过 10 MiB。")
    result = await asyncio.to_thread(
        get_agent().classify_file,
        content,
        filename,
        top_k=top_k,
        confidence_threshold=confidence_threshold,
    )
    return _result_response(result)


@app.post("/api/classify/file/stream")
async def classify_file_stream(
    file: UploadFile = File(...),
    top_k: int = Query(default=5, ge=1, le=110),
    confidence_threshold: float | None = Query(default=None, ge=0, le=1),
) -> StreamingResponse:
    """Stream document parsing and classification progress without saving uploads."""
    filename = file.filename or ""
    if Path(filename).suffix.lower() not in SUPPORTED_SUFFIXES:
        raise HTTPException(status_code=422, detail="仅支持 TXT、PDF 和 DOCX 文件。")
    try:
        content = await file.read(MAX_UPLOAD_BYTES + 1)
    finally:
        await file.close()
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="上传文件不能超过 10 MiB。")
    return _stream_response(
        lambda on_step: get_agent().classify_file(
            content,
            filename,
            top_k=top_k,
            confidence_threshold=confidence_threshold,
            on_step=on_step,
        )
    )


@app.post("/api/agent/run")
async def run_agent(req: AgentRequest) -> JSONResponse:
    """Keep the generic member-4 entry point for scripted integrations."""
    result = await asyncio.to_thread(get_agent().run, req.user_input)
    return _result_response(result)


@app.get("/api/statistics")
def statistics() -> dict[str, Any]:
    """Return the existing preprocessing reports in a browser-friendly shape."""
    statistics_report = _load_json(PROCESSED_DIR / "statistics.json")
    quality_report = _load_json(PROCESSED_DIR / "quality_report.json")
    split_report = _load_json(PROCESSED_DIR / "split_report.json")
    categories = [
        {"code": code, **entry}
        for code, entry in statistics_report.get("category_quality", {}).items()
        if isinstance(entry, dict)
    ]
    categories.sort(key=lambda item: (-int(item.get("count", 0)), str(item["code"])))
    return {
        "dataset": {
            "total": statistics_report.get("total", 0),
            "label_count": statistics_report.get("label_count", 0),
            "text_length_histogram": statistics_report.get("text_length_histogram", {}),
        },
        "split": {"train": split_report.get("train_count", 0), "validation": split_report.get("val_count", 0), "test": split_report.get("test_count", 0)},
        "quality": {
            "valid_before_dedup": quality_report.get("valid_before_dedup", 0),
            "cleaned_count": quality_report.get("cleaned_count", 0),
            "duplicates_removed": quality_report.get("duplicates_removed", 0),
            "rejected_count": quality_report.get("rejected_count", 0),
            "possibly_truncated_abstracts": quality_report.get("possibly_truncated_abstracts", 0),
        },
        "categories": categories,
    }


@app.get("/api/metrics")
def metrics() -> dict[str, Any]:
    """Serve saved evaluation artifacts; dashboard reads never retrain a model."""
    models: list[dict[str, Any]] = []
    matrices: dict[str, Any] = {}
    for model_name in ("medbert_v20", "macbert_v20"):
        models.append({"id": model_name, **_load_json(RESULTS_DIR / f"{model_name}_test_metrics.json")})
        matrices[model_name] = _load_json(RESULTS_DIR / f"{model_name}_test_confusion_matrix.json")
    return {"models": models, "confusion_matrices": matrices}


frontend_dist = PROJECT_ROOT / "frontend" / "dist"
if frontend_dist.exists():
    app.mount("/", StaticFiles(directory=frontend_dist, html=True), name="frontend")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("backend.main:app", host=WEB_CONFIG["host"], port=WEB_CONFIG["port"], reload=True)
