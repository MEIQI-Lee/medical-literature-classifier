# 对接验证记录

日期：2026-09-04。环境：macOS arm64，Python 3.12，CPU；PyTorch 2.7.1、Transformers 4.53.2、NumPy 1.26.4。

## 执行结果

| 检查 | 结果 |
|---|---|
| `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 ACADEMIC_REAL_MODEL_TESTS=1 python -m pytest -q` | **96 passed in 7.02s** |
| `python -m examples.member5_client --top-k 3` | 实际本地 MedBERT 成功预测，返回 ToolResult 和步骤事件 |
| `python -m agent --team --device cpu --top-k 1 --paper-json examples/sample_paper.json` | 核心 CLI 真实模型入口验证 |
| `python -m agent.offline check` | 已有部署包有效，`training_skipped=true` |
| `python -m pip check` | 无依赖冲突 |

96 项包含 90 项常规测试和 6 项显式启用的真实权重测试。默认 `pytest` 跳过真实权重测试，可供没有模型文件的页面开发环境使用；仓库数据和预处理文件仍需保留。

Ruff 检查成员4代码。原样合入的其他成员代码通过 `extend-exclude` 排除，没有为满足风格检查修改其他成员实现。

## 真实模型验证

- 直接使用 `model/checkpoints/medbert_v20/best`。
- 分类头对应 110 类；部署版本 `medbert_v20-705031ec3a53`。
- 使用成员3训练 `_collate()` 生成输入并执行模型，适配器全部 110 类分数与对照一致（绝对容差 1e-7）。
- 验证 Top-1 正常分支、低置信度至少 3 候选、复用同一模型对象。
- 超过 512 tokens 的输入正确报告截断与原始 Token 数。
- TXT、含文本层 PDF、DOCX 均完成“解析→预处理→真实推理→统一结果”。
- 心血管研究示例得到 `R54 / 心脏、血管（循环系）疾病`，原始概率 `0.944318950176239`。样例保存为 `docs/member5-real-result.json`。

## 兼容与隔离

- 两种 `get_agent` 导入路径、旧工具注册表、ToolResult 外层结构兼容。
- 不同并发请求的候选数量、阈值、request ID 和执行步骤保持独立。
- 并发首次使用只创建一次核心模型服务。
- 字段文本与成员2拼接规则一致；文件字段提取保留换行。
- 分类文本包含“训练”等词时不误调用离线训练。
- 采集测试只模拟上游函数，验证参数换算和返回结构，未访问知网。
- 已有权重检查不会调用采集、清洗或训练工具。

## 文件边界

与用户上传的 `academic-classification-agent(2)` 比较，以下内容逐文件一致：`backend/`、`frontend/`、`presentation/`、`model/`、`preprocess/`、`data/`。比较忽略 Python 缓存和 `.DS_Store`。

数据指纹差异仅作为元信息保留。没有重新训练、重建数据或重算整套测试集指标。没有运行成员5尚待完成的 Web API、页面或实时网络事件通道。CPU 联调通过不代表已验证 CUDA/MPS 环境或 OCR。
