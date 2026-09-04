> 本文保留核心设计与通用扩展协议。本次 MedBERT 对接以 `member4-integration-implementation.md` 为准；成员5使用 `member5-integration.md` 中的 ToolResult 服务接口。

# 组员对接协议 v1

本协议用于单标签分类。一个模型标签对应一个 CLC 类别，Top-K 是候选排序，不表示一篇文献同时具有 K 个真实标签。项目切换成多标签任务时，需要一起调整训练损失、输出语义和 Agent 概率校验。

## 成员 1、2：类别和标签文件

`category_mapping.json` 的格式如下。此处编码是占位符，正式文件由成员 1 提供准确的 CLC 编码和名称。

```json
{
  "version": "clc-r-v1",
  "categories": [
    {"code": "<CLC编码A>", "name": "<完整类别名称A>", "path": ["医药、卫生", "<父类别>", "<类别A>"]},
    {"code": "<CLC编码B>", "name": "<完整类别名称B>", "path": []}
  ]
}
```

`labels.json` 用明确的映射保存模型标签。`label_id` 可以不连续，但不能重复；同一类别不能对应两个标签。CLC 编码始终是字符串。

```json
{
  "version": "labels-v1",
  "category_version": "clc-r-v1",
  "labels": [
    {"label_id": 0, "category_code": "<CLC编码A>"},
    {"label_id": 1, "category_code": "<CLC编码B>"}
  ]
}
```

示例只展示结构。真实交付需覆盖训练任务的全部 100 个以上类别。不要根据 JSON 数组的位置临时推测 label_id；训练侧导出显式标签表，模型权重与标签表一起保存。

成员 2 的预处理器工厂约定：

```python
from agent.schemas import ParsedDocument, PreparedText


class TeamPreprocessor:
    version = "team-preprocess-v1"

    def prepare(self, document: ParsedDocument) -> PreparedText:
        # 这里复用训练数据构建时的同一套清洗、字段拼接函数。
        # document.fields 为显式输入的 title/keywords/abstract，可能为 None。
        # document.text 是从输入或文件提取的文本。
        text = your_shared_prepare_function(document)
        return PreparedText(text=text, strategy="free_text")


def create_preprocessor():
    return TeamPreprocessor()
```

这是一段接入示意，`your_shared_prepare_function` 需要替换为成员 2 的实际实现。`strategy` 应按实际处理方式取 `free_text`、`structured_fields`、`document_fields` 或 `document_fulltext`。预处理方法应保持无请求共享状态，以便并发调用。

成员 2 和成员 3 可直接采用默认 `builtin-v1`：清洗后按“标题、关键词、摘要”顺序拼接，空字段省略。若采用自己的清洗和拼接方式，必须注入自己的预处理器并使用对应版本；不能仅修改版本名来跳过检查。

## 成员 3：Python 模型接口

推荐提供以下结构，具体模型架构和深度学习依赖保留在成员 3 的模块中：

```python
def create_classifier(checkpoint_path: str):
    # 此处加载已训练好的分类器、Tokenizer/向量器及模型元信息。
    # 深度学习模型在这里设为推理模式，predict 中关闭梯度。
    return YourClassifier(checkpoint_path)


class YourClassifier:
    def describe(self) -> dict:
        return {
            "model_version": "classifier-v1",
            "label_version": "labels-v1",
            "preprocess_version": "team-preprocess-v1",
            "label_ids": [0, 1],
            "score_type": "probability",
            "calibrated": False,
            "mode": "model",
        }

    def predict(self, text: str, top_k: int = 5) -> dict:
        # 先对全部类别进行预测，再截取最高的 min(top_k, 类别数) 项。
        # Tokenizer 和截断逻辑由该方法负责。
        return {
            "model_info": self.describe(),
            "predictions": [{"label_id": 0, "score": 0.8}, {"label_id": 1, "score": 0.2}],
            "truncated": False,
            "input_tokens": 120,
        }
```

上面是两类的结构示意，并非模型实现。实际模型返回所有标签的 ID 集合；某次预测的候选数须等于 `min(top_k, 类别总数)`。返回顺序可以由模型排好，Agent 仍会按分数降序、同分时按 label_id 升序整理。

- `probability`：每项在 0～1，Top-K 总和不超过 1；返回全部类别时总和必须约为 1。不要对截取的 K 项重新 softmax/归一化。
- `decision_score`：例如 SVM 的决策值，可为负数，不要求和为 1。Agent 只展示分数和候选，不生成百分比置信度。
- 禁止 `NaN`、无穷大、重复标签、未知标签；模型信息必须与 `describe()` 一致。
- `calibrated` 仅在完成实际概率校准后设为 true。模型标签表和预处理规则变更时更新版本。
- 单标签任务中的 logits 转概率应在模型侧完成；Agent 不猜测数组是 logits、概率还是决策分数。

示例输出 JSON Schema 见 `docs/model-output.schema.json`。可直接返回字典或对应的 Pydantic 对象。

## 成员 3：脚本接口

脚本在配置的本机/服务器环境执行。Agent 使用参数数组启动进程，JSON 通过 stdin 传递，不通过 shell 拼接文本。

初始化时输入：

```json
{"action": "describe", "checkpoint_path": "/absolute/path/to/checkpoint"}
```

脚本 stdout 返回上一节 `describe()` 对应的 JSON 对象。

推理时输入：

```json
{"action": "predict", "checkpoint_path": "/absolute/path/to/checkpoint", "text": "已整理的文本", "top_k": 5}
```

脚本 stdout 返回上一节 `predict()` 的完整 JSON 对象。stdout 只能有一个对象；调试信息、加载进度和普通日志写到 stderr。失败时使用非零退出码。

`describe` 和 `predict` 都只加载已有权重，不能触发训练。对于已有的命令行脚本，可增加一个很薄的包装脚本转换输入输出。参考可执行示例 `examples/model_script_demo.py`，该示例始终标记 `mode="demo"`。

单次脚本接口会在每次请求时创建进程，不能承诺模型常驻内存。脚本默认超时 60 秒，可配置。Python 方式没有强制终止线程的超时机制；需要硬超时或进程隔离时选脚本/独立模型服务。

## 成员 5：返回结果与展示

`agent.classify_text(text)`、`agent.classify_file(bytes, filename)` 和 `agent.classify_paper(...)` 返回 `AgentResult`。通过 `.model_dump(mode="json")` 转成后端可序列化字典。完整定义见 `docs/agent-result.schema.json`。

| 字段 | 展示/处理方式 |
|---|---|
| `schema_version` | 当前协议版本为 `1.0` |
| `request_id` | 本次请求与步骤记录关联 ID |
| `status` | `success` / `error`，低置信度仍是 success |
| `mode` | `demo` 时页面明确显示“演示模式” |
| `prediction` | 首选候选；失败时为 null |
| `candidates` | Top-K 排序结果，包含编码、名称、路径、分数和 confidence |
| `confidence_status` | `normal` / `low` / `unknown` |
| `display_mode` | `top1` 突出首选；`candidates` 突出多个候选；`error` 展示错误 |
| `model_info` | 模型、标签及预处理版本；分数类型；是否经过校准 |
| `input` | 类型、字符数、提取策略、token 数和截断标记，不含正文 |
| `steps` | 每个执行节点的最终状态、耗时和摘要 |
| `warnings` | 文本截断、部分页面无法提取、演示模式等可读提示 |
| `message` | 可直接展示的结果说明 |
| `error` | 出错时的 code、message、step；成功为 null |

`prediction` 即候选列表第一项；当置信度偏低或无法判断时，它只是当前排名第一的备选，不表示已确认分类。`confidence` 非 null 时范围为 0～1，页面格式化时再乘 100。`score_type="decision_score"` 时 confidence 为 null。

实时步骤接口：

```python
events = []
result = agent.classify_text(text, on_step=events.append)
```

回调收到 `StepInfo` 对象，每一步先产生 running，再产生 success 或 error，两次事件具有相同 `sequence`。最终 `steps` 只保存每一步的终态。`sequence` 从 1 开始，`duration_ms` 单位毫秒。回调异常不会取消分类，结果会增加通知失败提示。

后端启动时捕获 `AgentError` 以报告配置/模型不可用；进入一次请求后，预期业务错误封装在 `AgentResult.error`。HTTP 状态码由成员 5 映射，例如输入错误 400、文件过大 413、模型不可用 503、脚本超时 504。前端用错误码判断逻辑，不匹配中文文案。

| 常见错误码 | 含义 |
|---|---|
| `EMPTY_TEXT` / `TEXT_TOO_SHORT` / `TEXT_TOO_LONG` | 文本为空、太短或超限 |
| `EMPTY_FILE` / `FILE_TOO_LARGE` / `TOO_MANY_PAGES` | 文件大小/页数异常 |
| `UNSUPPORTED_FILE_TYPE` / `INVALID_FILE` | 文件格式不支持或内容与格式不符 |
| `NO_EXTRACTABLE_TEXT` / `ENCRYPTED_PDF` / `DOCUMENT_PARSE_FAILED` | 文档提取失败 |
| `MODEL_NOT_READY` / `MODEL_INFERENCE_FAILED` | 模型初始化或推理失败 |
| `LABEL_VERSION_MISMATCH` / `PREPROCESS_VERSION_MISMATCH` | 模型与数据处理约定不一致 |
| `INVALID_MODEL_OUTPUT` / `MODEL_VERSION_MISMATCH` | 当前推理返回不符合协议 |
| `TOOL_TIMEOUT` / `INVALID_TOOL_OUTPUT` / `TOOL_EXECUTION_FAILED` | 脚本超时、协议错误或执行失败 |

## 置信度配置

成员 3 在独立验证数据上确定阈值后，将下列结构放入配置的 `agent.confidence`。数值必须来自实际验证；以下仅示意字段。

```json
{"model_version": "classifier-v1", "min_probability": 0.65, "min_margin": 0.12}
```

只有 Top-1 概率和 Top-1/Top-2 分差均达到阈值时才进入 normal 分支。阈值与模型版本绑定，版本不匹配时拒绝启动。原始决策分数不能使用概率阈值。没有配置阈值时 status 为 unknown，不悄悄采用演示参数。

## 成员 1～3：离线脚本接口

三种脚本从 stdin 接收 `{"action": "collect|clean|train", "checkpoint_path": "..."}`；成功时向 stdout 输出 `{"status": "success"}`，并实际生成配置里 `outputs` 约定的文件。模型目录参数用于训练输出，采集/清洗脚本可忽略它。

脚本自身从共同的项目配置读取其余参数；下游通过约定路径读取上游产物。当前编排检查输出存在且非空，数据数量、类别覆盖和模型质量由各成员自己的统计/评估代码负责。当前流程不是数据库事务，也不自动回滚脚本的部分输出。训练脚本应先写临时目录，完成后再发布可加载权重，避免残留半成品被下一次“已有权重”检查误判。

默认已有权重就跳过训练。开启数据准备时可能更新数据和标签文件，应另存版本并保持已部署模型的标签文件不变。正式上线前让模型加载器实际验证权重格式，文件存在检查本身无法证明权重有效。
