# 学术文本分类系统：成员 4 对接完成版

当前工程已合入成员 1—3 的数据、预处理代码、训练代码及 MedBERT v20 微调权重，并完成成员 4 的真实模型对接。实际部署为 110 类，清洗后数据 5450 条。默认使用已有权重进行预测。

本地验证：Python 3.12 / macOS arm64 / CPU，96 项测试通过，其中包含 6 项真实模型测试。没有重新训练、采集或修改数据集。

成员 5 的 Web 集成已完成：`backend/main.py` 提供分类、上传、统计与指标 API，`frontend/dist/` 提供无需 Node 构建的静态单页界面。项目展示 PPT 可按课程展示需要另行制作。

## 运行

在项目根目录执行。当前电脑的 `.venv` 已准备好，可直接使用 `.venv/bin/python`。换电脑应重新创建虚拟环境，建议 Python 3.12：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-model.txt
python -m agent --team --device cpu --paper-json examples/sample_paper.json
python -m agent --team --file examples/sample.txt --top-k 3 --steps
```

Windows 使用 `.venv\Scripts\Activate.ps1` 激活。CUDA 环境请按实际 CUDA 版本安装匹配的 PyTorch；本次仅验证 CPU。

给成员 5 的完整返回示例：

```bash
python -m examples.member5_client
python -m examples.member5_client --file examples/sample.txt --top-k 3
```

没有权重时可显式运行 `python -m examples.member5_client --demo` 做页面联调。Demo 使用虚构类别，返回中会标明 `mode=demo`，不会伪装成真实预测。

## 对外入口

```python
from agent import get_agent

service = get_agent()  # 每个后端进程复用；首次分类时加载模型
result = service.classify_text("分析高血压患者的心血管危险因素和临床治疗效果。", top_k=5)
payload = result.to_dict()  # ToolResult 格式：success/data/error/steps/logs

# 上传内容由成员5从 Web 框架读取：
# payload = service.classify_file(content, filename, top_k=5).to_dict()

# 可选：启动阶段提前加载，用于检查模型是否就绪。
# service.warmup()
```

分类是同步调用；异步后端应放在线程池中。`on_step` 同步回调提供字典事件，可由成员 5 接到自己的 SSE / WebSocket 通道。详见接入说明。

## Web 分类网站

先安装部署模型依赖，再启动 FastAPI：

```bash
python -m pip install -r requirements-model.txt
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

打开 `http://127.0.0.1:8000` 即可使用。前端由后端直接托管，支持文本输入与 TXT、PDF、DOCX 上传（最大 10 MiB）。页面会显示 Top-K 候选、模型分数、智能体步骤，以及已有的数据统计与模型历史评估结果。

| 路由 | 用途 |
| --- | --- |
| `POST /api/classify` | 文本分类，JSON：`text`、`top_k`、可选 `confidence_threshold` |
| `POST /api/classify/file` | 上传 TXT / PDF / DOCX，查询参数：`top_k` |
| `POST /api/agent/run` | 保留通用 Agent 服务入口 |
| `GET /api/health` | 进程状态；加 `?check_model=true` 可检查模型加载 |
| `GET /api/statistics` | 已生成的数据质量、分布与划分报告 |
| `GET /api/metrics` | MedBERT / MacBERT 历史测试指标和混淆矩阵数据 |

Web 框架依赖写在 `requirements-web.txt`，并由 `requirements.txt` 自动引入。在线分类需要 `requirements-model.txt` 中的 PyTorch 与 Transformers；首次模型加载会耗时，页面会保持“正在分类”状态。

底层 `create_medbert_agent()` 返回 `ClassificationAgent`，结果为 Pydantic `AgentResult`；上层 `get_agent()` 返回兼容服务，结果为 `ToolResult`。成员 5 统一使用后者即可。

## 已实现的成员 4 模块

| 模块 | 内容 |
|---|---|
| `agent/service.py` | 兼容 `get_agent()`、分类服务、任务分发、延迟加载 |
| `agent/agent.py`、`graph.py` | 输入校验、解析、预处理、分类、置信度分支、类别查询、结果生成 |
| `agent/integrations/team_preprocessor.py` | 复用成员2的清洗和字段拼接；保留训练使用的字段标点 |
| `agent/integrations/medbert_backend.py` | 复用成员3的模型对象；完整精度分数、Token 数和截断信息 |
| `agent/integrations/team_contract.py` | 结果和步骤转换为旧工程兼容格式 |
| `agent/tools/` | 文档解析、分类、类别查询，以及已有清洗/训练工具和爬虫适配 |
| `agent/offline.py` | 显式离线任务入口；默认只检查已有模型 |
| `examples/member5_client.py` | 可运行的真实模型 / Demo / 文件调用示例 |

`model/`、`preprocess/`、`crawler/`、`data/`、`results/` 来自成员 1—3。数据指纹差异只在 `warmup()` 和离线检查的元信息中记录，不阻塞分类，也不触发重训。

## 输入与结果规则

- 模型固定为 `model/checkpoints/medbert_v20/best`，标签以该 checkpoint 自带文件为准。换位置可使用 `AcademicClassificationAgent(checkpoint_path=..., device=...)`。
- 默认 `top_k=5`；支持每次请求单独传正整数，最多返回模型全部 110 类。
- 默认概率阈值沿用上游的 0.6，属于经验阈值；暂不启用额外的分数差阈值。低置信度或未启用判断时至少提供 3 个候选（受实际类别数限制）。
- `requested_top_k` 和 `returned_top_k` 分别记录请求和实际返回数量；低置信度时请求 Top-1 也可能返回 3 项。
- 分数是全部类别 Softmax 后的原始概率，Top-K 不重新归一化；`calibrated=false`，不等于真实正确率。
- 最大输入长度为模型 Tokenizer 的 512 tokens，包含特殊 token；`input_tokens` 是截断前数量，超长时返回 `truncated=true` 和提示。
- TXT / PDF / DOCX 默认最多 10 MiB；PDF 最多 100 页；有效文本 10—200000 字符。PDF 需要文本层，目前没有 OCR。

## 离线工具

```bash
# 默认只检查已有模型。不会采集、清洗或训练。
python -m agent.offline
```

只有准备重新生成数据或训练新版本时才使用以下入口，并安装 `requirements-offline.txt`：

```bash
python -m agent.offline collect --categories R51 R52 --target-count 100
python -m agent.offline clean
python -m agent.offline train --output-name medbert_v21 --epochs 20 --device cuda
```

`train` 是显式训练新版本的命令，禁止覆盖 `medbert_v20` 或已有输出目录。新的训练结果不会自动切换成在线部署模型。

现有 `agent.pipeline` 和 `config/*.example.json` 继续作为通用脚本协议模板；当前团队对接直接使用上述入口。`collect`、`clean`、`train` 会写入数据或模型目录，本次验证没有执行这些真实任务。

## 依赖与验证

- `requirements.txt`：核心 Agent 与文件解析依赖。
- `requirements-model.txt`：加上本次验证的 PyTorch 2.7.1、Transformers 4.53.2、NumPy 1.26.4。
- `requirements-offline.txt`：额外的采集、训练和评估依赖。
- `requirements-lock.txt`：本次本地虚拟环境的完整版本快照。
- `docs/upstream/requirements-original.txt`：上游原始依赖，仅供参考；不要覆盖当前依赖。
- Web 框架与前端依赖由成员 5 维护。

```bash
python -m pip install -e '.[dev]'
python -m pytest -q
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 ACADEMIC_REAL_MODEL_TESTS=1 python -m pytest -q
ruff check .
python -m pip check
```

默认测试跳过 6 项真实模型测试。Ruff 只检查成员 4 的代码，原样保留的其他成员文件不在本次格式检查范围内。验证细节见 [docs/validation.md](docs/validation.md)。

交付给组员时带上源代码、数据和部署权重；不要复制 `.venv`。权重已在当前目录，但由 `.gitignore` 排除，单独通过 Git 传代码时需另传模型文件。
