# 成员 5 接入协议

此文档描述已经实现的成员 4 Python 服务。后端路由、HTTP 文件读取、前端页面和可视化由成员 5 实现。不要将 `backend/main.py` 目前的占位返回误认为已经完成的分类 API。

## 服务入口

```python
from agent import get_agent

service = get_agent()
# 可选启动预热：异常为 AgentError，由后端决定就绪状态。
metadata = service.warmup()

result = service.classify_text(
    text="分析高血压患者的心血管危险因素与治疗效果。",
    top_k=5,
    confidence_threshold=0.6,
)
response = result.to_dict()
```

`get_agent()` 本身不加载模型，第一次分类或 `warmup()` 才加载。每个后端进程只有一个默认服务实例。多 worker 会各自占用模型内存；课程演示建议先使用单 worker。

可选独立实例：

```python
from agent import AcademicClassificationAgent

service = AcademicClassificationAgent(device="cpu", trace_path="logs/classifications.jsonl")
```

JSONL 日志记录结果与步骤，默认不写原始输入文本。路径由服务端配置。

| 方法 | 输入 | 返回 |
|---|---|---|
| `classify_text(text, *, top_k=None, confidence_threshold=None, on_step=None)` | 普通文本或明确标注的标题/关键词/摘要 | `ToolResult` |
| `classify_file(content, filename, *, top_k=None, confidence_threshold=None, on_step=None)` | `bytes` 与文件名；支持 TXT/PDF/DOCX | `ToolResult` |
| `classify_paper(*, title='', keywords=None, abstract='', top_k=None, confidence_threshold=None, on_step=None)` | 结构化字段；关键词支持字符串或字符串列表 | `ToolResult` |
| `warmup()` | 无 | 模型信息与部署信息字典；失败抛出 `AgentError` |
| `run(user_input)` | 文本或结构化任务字典 | `ToolResult` |

上层服务返回 `ToolResult`，序列化使用 `.to_dict()`。底层 `ClassificationAgent` 才使用 Pydantic `.model_dump()`，不要混用。

## 后端路由连接点

| 原有路由 | 成员5需要连接的内容 |
|---|---|
| `POST /api/classify` | 请求 `text`、`top_k`，调用 `service.classify_text()` |
| `POST /api/classify/file` | 读取上传内容与文件名，调用 `service.classify_file()` |
| `POST /api/agent/run` | 保留 `service.run(req.user_input).to_dict()`；结构化任务可以另扩展请求模型 |
| `GET /api/health` | 自行区分进程可用与模型就绪；可用 `warmup()` 做启动检查 |
| `GET /api/statistics` | 读取已有数据报告，转换为页面所需结构 |
| `GET /api/metrics` | 读取已有模型实验 JSON，处理图表数据 |

`ClassifyTextTool(file_path=...)` 不负责直接读取服务器路径；上传统一走 bytes 接口，不必将文件落盘。

服务调用是同步、可能耗时的操作。如果后端路由用 `async def`，应在线程池调用。以下只是接入片段，不是已经实现的路由：

```python
from starlette.concurrency import run_in_threadpool

# file 是 UploadFile，成员5负责 multipart 接收和关闭文件。
content = await file.read(10 * 1024 * 1024 + 1)
result = await run_in_threadpool(
    service.classify_file,
    content,
    file.filename or "",
    top_k=5,
)
response = result.to_dict()
```

读取上限多一个字节是为了让 Agent 返回文件超限错误。Web 层也应设置上传限制。同步 `def` 路由可直接调用服务。

## 返回结构

统一外层：

```text
success: boolean
error: string | null
data: object
steps: [{step_id, name, status, detail, timestamp}]
logs: string[]
```

分类成功时 `data`：

| 字段 | 含义 |
|---|---|
| `request_id` | 每次请求独立的标识 |
| `prediction.top1` | `{code, name, confidence}` 首选类别 |
| `prediction.topk` | 同结构候选数组，按分数降序排列 |
| `prediction.model_type` | `medbert_v20` |
| `prediction.checkpoint` | 保留旧接口的部署路径字段；页面一般无需显示 |
| `low_confidence` | 是否低于当前判断策略 |
| `confidence_status` | `normal` / `low` / `unknown` |
| `confidence_threshold` | 本次实际使用的概率阈值 |
| `display_mode` | `top1` / `candidates` / `error` |
| `requested_top_k`、`returned_top_k` | 请求数量和实际候选数量 |
| `message` | 总体提示 |
| `model_info` | 模型、标签和预处理版本；`mode` 与 `calibrated` 等 |
| `input` | 文件名、来源类型、提取长度、Token 数、截断情况 |
| `warnings` | `{code, message}` 提示数组 |
| `duration_ms` | 工作流耗时，不含首次模型加载 |
| `error_code` | 成功为 null，失败为错误编码 |

真实返回见 `member5-real-result.json`；页面联调用 Demo 见 `member5-demo-result.json`；错误示例见 `member5-error-result.json`。`tool-result.schema.json` 描述通用外层结构，内部业务字段以本表和样例为准。

错误时必须先检查 `success`。进入工作流后的失败会保留已有步骤；初始化失败没有模型信息，`data` 仅保证错误编码和请求标识，不能直接解引用 `data.prediction.top1`。

常见错误：`EMPTY_TEXT`、`TEXT_TOO_SHORT`、`INVALID_INPUT`、`FILE_TOO_LARGE`、`UNSUPPORTED_FILE_TYPE`、`NO_EXTRACTABLE_TEXT`、`MODEL_NOT_READY`、`MODEL_INFERENCE_FAILED`。HTTP 状态码由后端决定；例如输入问题 400/422、文件过大 413、模型未就绪 503。

## Top-K 与页面展示

默认 `top_k=5`，正整数均可，最多返回模型全部 110 类。`normal` 返回请求数量；`low` 或 `unknown` 至少保留 3 个候选。例：请求 1、低置信度时 `requested_top_k=1`，`returned_top_k=3`。页面使用实际数组长度渲染。

默认阈值为 0.6，是沿用上游工具的经验配置。当前输出未校准，应显示“模型置信度/候选分数”，不要写成“预测正确率”。`normal` 也不意味着保证分类正确；超出模型医学类别范围的输入未单独训练拒识模型。

Token 数为截断前数量，包含特殊 token。大于 512 时 `input.truncated=true`，展示 `warnings` 中的截断提示。扫描版 PDF 返回需要 OCR 的提示，目前没有 OCR 工具。

## 执行过程

```python
events = []
result = service.classify_text(text, on_step=events.append)
```

回调收到字典，状态为 `running` / `done` / `failed`。同一步开始与结束事件的 `step_id` 相同，前端可以更新同一行；最终 `result.steps` 保存每一步的结束状态。

节点依次为：`validate_input` → `read_text` 或 `parse_document` → `prepare_text` → `classify_text` → `select_candidates` → `accept_prediction` 或 `review_candidates` → `query_category` → `generate_result`。

同步回调在执行分类的线程触发。SSE/WebSocket 由成员 5 实现；可通过线程安全队列或 `loop.call_soon_threadsafe()` 将事件送回异步事件循环。首次加载发生在这些节点之前，需要“模型加载中”状态时由后端预热流程提供。

## 数据与实验图表

直接读取文件，不要在统计接口中调用清洗或重新训练：

| 文件 | 用途 |
|---|---|
| `data/processed/statistics.json` | 类别分布、文本长度和数量统计 |
| `data/processed/quality_report.json` | 原始数据与清洗情况 |
| `data/processed/split_report.json` | train/val/test 划分 |
| `data/labels.json` | 当前数据标签 |
| `category_mapping.json` | 128 类目录；只有 110 类由当前模型支持 |
| `results/medbert_v20_test_metrics.json` | MedBERT 历史测试指标 |
| `results/macbert_v20_test_metrics.json` | MacBERT 历史对比指标 |
| `results/*_test_confusion_matrix.json` | 混淆矩阵图表数据 |

实际数据为 5450 条、110 类，划分为 3811/547/1092。旧的成员2文档中 124 类等数字不作为页面常量。MacBERT 目前只有实验资料，不作为可切换部署模型。

实验文件内可能有 AutoDL / Windows 历史绝对路径，统一从当前项目根目录解析文件。混淆矩阵 PNG 未随工程提供，可由前端根据 JSON 渲染。实验耗时来自原实验环境，不是当前网页响应速度。模型训练数据指纹差异仅记录在部署信息，不阻止调用。

## 离线任务与成员分工

普通分类页面只调用 `classify_text` / `classify_file` / `classify_paper`。`run()` 是任务分发入口，支持：

```python
# 示例；只有明确需要写入数据/训练时才调用这些离线任务。
service.run({"task": "collect_data", "categories": ["R51", "R52"], "target_count": 100})
service.run({"task": "clean_dataset"})
service.run({"task": "train_model", "model_type": "medbert", "output_name": "medbert_v21"})
service.run({"task": "query_category", "code": "R54"})
```

文本只有完全等于“训练/训练模型”“清洗/清洗数据/清洗数据集”等明确命令时才走离线任务；论文正文中出现这些词仍作为分类文本。离线长任务的 HTTP 调度、进度和访问入口由成员 5 决定，课程演示也可直接使用命令行。

`serve_web` 仍保留上游占位工具，网站启动由成员 5 的服务配置实现。当前 Agent 使用明确的 LangGraph 工作流，无外部 LLM/API Key 依赖。
