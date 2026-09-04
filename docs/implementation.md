> 本文保留核心设计与通用扩展协议。本次 MedBERT 对接以 `member4-integration-implementation.md` 为准；成员5使用 `member5-integration.md` 中的 ToolResult 服务接口。

# 成员 4 代码实现讲解

## 1. 这套 Agent 做什么

实现采用有状态、带条件分支的工具工作流。输入是文本、结构化论文字段或文档字节；输出是统一的分类结果与执行步骤。每个工具完成一个明确功能，工作流负责组织这些工具。

当前没有 LLM 调用、API Key 或自主规划提示词。LangGraph 节点可以是普通 Python 函数，图的节点执行工作，边决定下一步。这与所选工作流方案一致。[LangGraph Graph API](https://docs.langchain.com/oss/python/langgraph/graph-api)

后续如果课程明确要求 LLM 动态选择工具，可以在现有工具之上增加任务路由层；模型输出协议、预处理器和前端返回结构仍可保留。现在的分类标签始终来自分类器输出及类别表。

## 2. 从入口到结果

`ClassificationAgent` 构造时接收 `classifier`、`catalog`、`preprocessor`、`config` 和可选 `trace_sink`。这些对象分别代表模型、类别表、文本处理规则、参数和日志保存器。构造函数完成模型/标签/预处理版本检查后，编译工作流图。

调用 `classify_text()` 会构造 `InputRequest("text", text)`。文件和结构化字段入口构造各自类型的请求，然后统一进入 `_run()`。每次 `_run()` 都创建独立的 request_id、步骤列表和 warnings，避免后端连续或并发请求混用状态。

图的主要路径是：

```text
validate_input
  ├─ 文本/字段 → read_text
  └─ 文件      → parse_document
                    ↓
               prepare_text
                    ↓
               classify_text
                    ↓
             select_candidates
  ├─ normal  → accept_prediction
  └─ low/unknown → review_candidates
                    ↓
               query_category
                    ↓
              generate_result
```

任一步出错都会转到 `generate_result`，保留已执行步骤并生成统一失败结果。工作流图由 `graph.py` 定义，业务工具在 `tools/` 中，对外入口和工具节点桥接在 `agent.py` 中。

## 3. 工具是怎样封装的

本实现的 tools 是输入输出明确的 Python 函数或对象方法。因为调度由工作流负责，不需要给每个工具加 LLM 专用装饰器。

`parse_document(content, filename, config)` 接收内存中的文件字节，返回 `ParsedDocument`。后端无需把文件保存到 Agent 指定目录。文件名仅用于判断类型和展示，先取 basename，不把上传文件名作为落盘路径。

TXT 按 UTF-8、带 BOM 的 UTF-16 或 GB18030 读取；PDF 逐页提取并检查总页数；DOCX 先检查压缩包大小，再按正文中段落和表格的顺序提取。pypdf 无法自行对扫描图片做 OCR，因此无可提取文本时返回明确错误。[pypdf 文本提取说明](https://pypdf.readthedocs.io/en/stable/user/extract-text.html)

`DefaultTextPreprocessor.prepare(document)` 输出 `PreparedText`。默认规则做 Unicode 格式统一、换行/空白处理和常见 HTML 清理，保留数字、英文、医学缩写及比较符号；结构化字段按标题、关键词、摘要顺序组合。文件中有明确摘要标题时提取相应字段，否则使用全文，并记录 `FULLTEXT_FALLBACK`。

默认预处理器有版本 `builtin-v1`。真实模型可以复用它，也可通过配置导入成员 2 的预处理工厂。只有模型声明的 `preprocess_version` 与对象的 `version` 相等才允许启动。这样能在联调时尽早发现训练输入与在线输入不同的问题。

`classify_text()` 工具调用模型适配器，校验返回的标签、数量、版本和分数，再做排序。`select_candidates()` 根据配置判断置信度。`query_categories()` 将模型标签换成页面需要的编码和名称。结果节点使用模板生成简短提示，不编造文献结论或类别名称。

## 4. 为什么增加模型适配层

成员 3 的模型可能来自 sklearn，也可能来自 PyTorch 等框架。Agent 本身只依赖 `describe()` 和 `predict()` 协议，模型的具体依赖无需加到 Agent 核心代码里。

Python 模式由 `factory.py` 从配置导入工厂，传入已有权重路径。工厂初始化一次，`PythonModelAdapter` 持有返回的模型对象。每次预测复用该对象，并通过锁将模型推理串行化，便于避免多个请求同时占用同一模型的可变状态或 GPU 资源。

脚本模式使用 `ScriptModelAdapter`。初始化先发送 describe 请求，预测时发送 predict 请求。`subprocess.run` 使用参数列表、stdin JSON、超时与退出码检查；用户文本不会被当作 shell 代码。脚本不在另建的执行沙箱中运行，而是使用配置的解释器与工作目录。

脚本适配优先解决“同学已经封装为脚本”的兼容问题。它是每次请求一个进程，因此可能反复加载权重；需要较低延迟时使用 Python 方式，或后续实现常驻服务适配器。

## 5. 标签、版本和分数为什么要检查

模型可能预测 label_id=17，但页面需要的是某个具体 CLC 编码及名称。`CategoryCatalog` 保存两级映射：`label_id → category_code → Category`。构造时检查重复标签、重复编码、缺失类别及版本，再核对模型声明的完整标签集合。

预测输出也要检查：标签不重复、来自已知集合、返回数量正确、模型元信息未改变、分数有限。概率不能超过 0～1，Top-K 和不能大于 1；如果返回全部类别，和需要近似为 1。普通决策分数不做概率校验，也不转成 confidence 百分比。

当模型训练后换了一份标签映射，版本检查可以阻止“程序能跑但类别名称对应错了”。版本由小组共同维护，不能复用同一版本名表示不同标签内容。

## 6. 置信度分支

候选默认保留 Top-5。正常分支只改变 `display_mode`，突出 Top-1，并不丢弃其他候选。低置信度或无可用阈值时突出候选列表。

概率模型在配置了匹配版本的阈值后，用两个条件联合判断：`p1 >= min_probability`，且 `p1 - p2 >= min_margin`。任一不满足就是 low。无阈值或只有决策分数时为 unknown。不会在低置信度时再次调用同一个模型生成另一组分数。

返回值里的 probability/score 是模型输出，confidence_status 是规则判断，两者含义不同。未校准概率可能与实际正确率差异很大；校准应使用独立于模型拟合数据的预测结果。[scikit-learn 概率校准文档](https://scikit-learn.org/stable/modules/calibration.html)

演示分类器以关键词匹配产生 5 类分数：文本明确包含高血压、冠心病等词时走 normal；没有匹配词时各类分数相同，走 low。这是为了稳定验证分支，不能作为课程模型实验结果。

## 7. 执行步骤和日志

`_run_step()` 包装每个节点，负责发出 running 事件、计时、执行函数、捕获异常并发出终态事件。普通节点只返回状态更新和一句执行摘要，不需要重复写计时/异常代码。

最终 `steps` 保存每一步的 success/error 终态；`on_step` 能即时收到 running 和终态事件。每条记录带 request_id 和 sequence，成员 5 可以关联请求、排序并更新同一步的显示状态。

`JsonlTraceSink` 按行保存完整的公开结果，包括候选和步骤，不保存原始输入正文。写日志和事件回调失败不会取消已完成的分类，而是在 warnings 中说明。日志锁只协调单进程内线程，多后端进程应使用不同日志文件或成员 5 的统一日志设施。

## 8. 离线训练为何独立

在线入口没有训练分支。缺少权重就明确报错，不在用户上传文件后突然开始长时间训练。

`pipeline.py` 是另外一个命令行入口，仅按显式选项调用采集、清洗、训练脚本。默认已有权重时全部跳过；`--prepare-data` 选择采集和清洗；`--train-if-missing` 允许无权重时训练。每个阶段必须返回成功，并生成约定的非空文件，否则停止后续阶段。

该模块完成成员 4 的调度职责，实际采集规则、数据集划分、模型训练和评估继续由成员 1～3 实现。当前协议测试用临时脚本模拟三个阶段，未运行真实训练。

## 9. 如何验证与演示

可以依次展示：输入心血管相关文本、上传示例 TXT、输入主题不明确的文本、上传没有文字层的 PDF。四种场景分别体现自动分类、文件工具路由、低置信度分支和异常处理。

自动测试额外覆盖文档编码、DOCX 表格位置、PDF 部分空页、模型版本变化、异常分数、脚本超时、模型工厂复用、并发状态隔离和训练跳过逻辑。所有测试均可以在无 GPU、无真实权重、无知网访问的情况下运行。

真正接入后还需与成员 3 做一次同输入对照：使用同一预处理文本直接调用真实模型，再经过 Agent 调用，验证候选标签和分数一致。模型 Accuracy、Macro-F1、类别覆盖与现场推理速度必须基于真实交付物单独验证。
