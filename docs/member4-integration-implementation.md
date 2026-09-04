# 本次对接的代码实现

## 从整体看

成员 1—3 的文件被合入当前工程，原始上传目录保持不变。底层继续使用原有 `ClassificationAgent` 和 LangGraph，外部增加 `AcademicClassificationAgent` 兼容服务。网页端使用 `get_agent()`，不需要了解模型、Tokenizer 或 LangGraph 的内部结构。

本次的实际调用链：

```text
get_agent().classify_text/classify_file/classify_paper
    → service.py：取得共用的 ClassificationAgent
    → graph.py：执行请求独立的图状态
    → team_preprocessor.py：生成与训练一致的文本
    → PythonModelAdapter：串行调用推理
    → medbert_backend.py：现有模型的 tokenizer → forward → softmax
    → prediction.py：分数校验、置信度判断、类别补全
    → team_contract.py：生成 ToolResult
```

## 1. 服务兼容层：agent/service.py

`AcademicClassificationAgent` 提供上游同名类和 `get_agent()`。同时保留 `from agent.agent import get_agent` 和 `from agent import get_agent` 两种导入路径。

默认服务是进程内单例。创建服务不立即加载模型，`_get_core()` 在首次使用时调用 `create_medbert_agent()`。初始化锁保证多个并发首请求只创建一次核心对象。`warmup()` 可供后端启动阶段主动检查。

`classify_text()`、`classify_file()`、`classify_paper()` 都调用 `_classify()`，再统一转换为 `ToolResult`。如果初始化失败，仍返回带错误编码和失败步骤的结构，而不要求 Web 层处理所有内部异常。

原有 Agent 的 `history`、`trace` 会积累在共享实例里。当前服务不保留跨请求的可变历史，每次请求的状态、候选和步骤都放在 LangGraph 本次调用的 state 中。

`plan()` 只将完全匹配的离线命令映射为离线工具。普通学术文本即使包含“训练”“清洗”等词也继续分类。结构化 `run({task: ..., ...})` 让调用方明确指定工具，无须从长文本猜意图。训练默认参数改为实际支持的 `medbert`。

## 2. 预处理适配：team_preprocessor.py

`TeamPreprocessor.prepare()` 最终使用成员2的 `clean_special_chars()` 和 `build_text_input()`。结构化字段按“标题：… 。关键词：… 。摘要：…”的现有拼接规则输出，关键词内部使用顿号；不增加额外空格。

文件文本先识别明确的字段标题，再清洗内容，避免换行过早被折叠。既支持多行字段，也支持数据集中的单行串接形式。识别不完整时提示缺失字段；无法识别时使用全文并返回 `FULLTEXT_FALLBACK`。

模型适配器直接分词最终文本，不再次执行整段 NFKC。这保留了训练集构造函数插入的全角字段冒号。真实测试使用成员3训练代码中的 `_collate()` 生成对照输入，并比较全部 110 类分数，验证一致性。

`preprocess_version` 包含成员2清洗源文件的摘要，便于标识运行规则。这里只是版本记录，不重新生成训练数据。

## 3. 模型适配：medbert_backend.py

`inspect_bundle()` 检查微调模型必需的权重、Tokenizer、配置和 checkpoint 标签文件；空目录或只有 `.gitkeep` 的目录不能通过。标签 ID 必须连续，正反映射必须一致，模型配置中的标签也必须匹配。

模型版本由部署目录名与权重 SHA256 摘要构成。标签版本依据类别编号、编码和名称生成，与数据文件指纹独立。当前与训练数据指纹不同只体现在部署元信息，不阻塞分类。

`MedBertBackend` 调用成员3 `load_model()` 并复用其返回的模型和 Tokenizer；`model/` 源码与权重没有修改。推理保持 `eval()` 和 `torch.inference_mode()`，对所有类别进行 Softmax，随后排序取 Top-K，同分按 label ID 排序。

`predict()` 返回完整精度分数，避免上游展示函数中六位小数舍入对校验产生影响。Tokenizer 计算截断前 Token 数，超过 512 时按训练规则截断并标记。输出类别数量和维度仍会被验证。

`factory.create_medbert_agent()` 把模型适配器、类别表、成员2预处理器和概率阈值组合成完整核心。默认阈值 0.6、分数差阈值 0；模型概率标记为未校准。

## 4. 请求级参数与候选分支

`InputRequest` 新增 `top_k` 和 `confidence_threshold`。`validate_input` 在本次状态里建立候选数量和判断策略，不修改共享 `AgentConfig`。

内部至少获取 3 个候选（受类别数限制），用前两名执行已有的置信度逻辑。正常分支按请求数量返回；低置信度或无阈值分支至少保留 3 类。结果新增 `requested_top_k`、`returned_top_k`、`confidence_threshold`。

参数检查拒绝非正整数 Top-K、布尔值冒充数值、NaN/无穷大阈值。候选数量最多为实际模型类别数，Top-K 概率不重新归一化。

## 5. 输出转换：team_contract.py

底层 `AgentResult` 保持强类型，兼容层把类别字段映射到上游的 `data.prediction.top1/topk`，保留 `success/data/error/steps/logs`。

`to_tool_step()` 把核心 `running/success/error` 映射为上游 `running/done/failed`。步骤标识包含 request ID 和顺序号，因此同一步开始和结束事件能对应，多个请求也不会混淆。

除旧字段外，返回模型版本、输入摘要、截断提示、实际候选数和结构化错误编码。输入原文不会放进返回对象或默认 JSONL 日志。

`on_step` 在上层转换成普通字典，成员5可以直接序列化。通知失败只附加提示，不中断模型分类。

## 6. 已有工具的合并

`BaseTool`、`ToolResult`、`CleanDatasetTool`、`TrainModelTool` 保留上游实现。`ClassifyTextTool` 转为调用新服务，避免旧工具和核心各做一次置信度判断。

`CollectDataTool` 调用成员1已有的 `run_full_crawl()`。`target_count` 是期望总量，爬虫接受每类上限，因此按类别数量向上取整；结果同时返回请求总量、每类上限、实际配置上限和爬虫实际采集数量。爬取测试只验证参数桥接，没有访问知网。

`QueryCategoryTool` 读取类别目录和 checkpoint 标签，返回类别是否在当前部署模型范围内。查询无需加载模型。

`agent.offline` 默认只检查已有部署模型。采集、清洗和训练都需要明确子命令。`train` 要求新的输出名称，沿用成员3的防覆盖检查，不会自动切换部署版本。

## 7. 成员5继续实现的部分

`backend/`、`frontend/`、`presentation/` 原样保留。提供的 `examples/member5_client.py` 和 `examples/backend_integration.py` 只演示 Python 调用，没有实现 HTTP 路由。

成员5负责上传接收、API请求模型和状态码、异步线程池调度、SSE/WebSocket传输、页面与图表、统计/指标 JSON 的 Web 输出、服务部署及展示材料。接口文档、真实与 Demo 返回样例、通用外层 JSON Schema 已提供。

## 8. 验证范围

96 项测试包含之前的核心回归、上游预处理测试、本次兼容/并发测试，以及 6 项真实 MedBERT 测试。真实测试验证推理数据流和接口，不等于重新评估准确率。所有数据、权重和成员5目录已与源目录比较，内容一致。
