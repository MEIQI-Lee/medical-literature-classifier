# 成员 4 交接说明

本次基于 `academic-classification-agent(2)` 的成员 1—3 成果完成集成。成员 4 代码与全部已有数据、模型资产已整理到当前项目根目录。

## 交付边界

已完成：LangGraph 分类工作流；TXT/PDF/DOCX 解析；成员2预处理适配；成员3 MedBERT 推理适配；Top-K 与低置信度分支；类别名称补全；执行步骤回调；兼容 ToolResult 的服务入口；清洗/训练工具保留和采集工具连接；示例、协议与测试。

成员 5 继续完成：`backend/main.py` 的实际 HTTP 路由与上传处理、Web 依赖和启动配置、前端页面、过程可视化、统计和模型性能页面、演示 PPT。`backend/`、`frontend/`、`presentation/` 与上传工程逐文件一致，未实现或覆盖其中的 TODO。

## 成员 5 最短接入路径

1. 阅读 `docs/member5-integration.md`。
2. 执行 `python -m examples.member5_client`，观察真实返回和 stderr 中的步骤事件。
3. 在后端导入 `from agent import get_agent`。
4. 文本调用 `get_agent().classify_text(text, top_k=5).to_dict()`。
5. 文件调用 `get_agent().classify_file(content_bytes, filename, top_k=5).to_dict()`。
6. 按 `success` 判断结果，再读取 `data.prediction.top1` 和 `data.prediction.topk`。
7. SSE/WebSocket、HTTP 状态码和页面状态由 Web 层实现，参考文档中的对接约定。

已有 `/api/agent/run` 的 `get_agent().run(...).to_dict()` 调用兼容保留。分类按钮应调用明确的分类方法，不经过文本任务分发。

## 对接决定

- 在线模型固定 MedBERT v20 的微调 checkpoint，默认不训练。
- 训练数据指纹差异仅记录；类别编号和模型维度仍必须一致。
- 结构化字段采用成员2的 `build_text_input()`；适配器推理前不重复清洗。
- 0.6 置信度阈值沿用现有工具，不宣称已校准。低置信度至少返回 3 个候选。
- 所有请求参数和步骤独立；后端可以复用单例。模型推理通过锁串行执行。
- 保留原有 BaseTool、ToolResult、工具注册表、清洗工具、训练工具及 `get_agent()` 导入路径。
- `agent/prompts` 中的旧 LLM 模板继续保留；当前流程使用明确的 LangGraph 分支，没有接入外部 LLM，也不需要 API Key。

## 当前资产与验证

当前数据为 110 类 / 5450 条，train/val/test 为 3811/547/1092。`MEMBER2_HANDOFF.md` 中的旧数量仅属于历史记录，展示页应读取当前统计 JSON。

已安装并使用真实权重在 CPU 上验证。96 项测试通过，其中 6 项真实模型测试覆盖与训练 collator 的推理一致性、Top-K/阈值、长文本截断和三种上传格式。未执行实际采集、清洗重建或训练，也未重新评估完整测试集。

真实示例 `docs/member5-real-result.json` 中，心血管研究文本的首选为 `R54`，概率约 `0.944319`。这是一次样例预测，不是新增的准确率评估。

运行、依赖、目录说明见根目录 README。详细实现说明见 `docs/member4-integration-implementation.md`。
