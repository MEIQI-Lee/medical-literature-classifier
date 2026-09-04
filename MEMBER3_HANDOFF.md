# 阶段 3 交接：训练、评估与推理

阶段 3 已完成文本分类模型的训练、评估和推理模块；后续同学只需在 Agent Tool 中调用这些接口，无需修改 `model/` 内实现。

## 本阶段完成内容

- 基于 110 类、5,450 条清洗后数据，实现 MedBERT/MacBERT 的训练流程。
- 默认训练配置：MedBERT、20 epoch、batch size 8。
- 实现测试集评估：Accuracy、Precision、Recall、Macro-F1、Micro-F1、推理耗时和混淆矩阵。
- 实现默认 MedBERT v20 的 Top-K 推理；返回类别代码、类别名和置信度。
- 集成 AutoDL 的实验记录。MedBERT v20 在当时的测试划分上为 Accuracy `0.7106`、Macro-F1 `0.7036`。

## 更新文件

- `model/train.py`：训练。
- `model/evaluate.py`：评估。
- `model/classifier.py`：模型加载与推理。
- `model/metrics.py`：指标与混淆矩阵。
- `model/checkpoints/medbert_v20/best/`：当前默认微调模型的配置、tokenizer、标签映射和本地权重。
- `results/`：MacBERT v20、MedBERT v20 的实验结果。
- `config.py`：默认模型与训练参数。

## 提供给 Tool 的接口

```python
from model.classifier import classify
from model.train import train_model
from model.evaluate import evaluate_model

# 默认 MedBERT v20 推理
prediction = classify(text, top_k=5)

# 训练新版本
training = train_model(output_name="medbert_v21")

# 评估新版本
metrics = evaluate_model(
    checkpoint=training["checkpoint"],
    output_name="medbert_v21_test",
)
```

`classify()` 返回：

```python
{
    "top1": {"code": "R599", "name": "类别名", "confidence": 0.98},
    "topk": [...],
    "model_type": "medbert_v20",
    "checkpoint": ".../medbert_v20/best",
}
```

## Tool 封装状态

- `agent/tools/clean_dataset.py` 已由阶段 2 接通。
- `agent/tools/train_model.py` 已接通：训练新版本后自动评估，并按 `BaseTool` 返回 `ToolResult`。
- `agent/tools/classify_text.py` 已接通：直接调用 `classify(text, top_k=top_k)`，无需重复加载模型或查询类别名称；文件解析仍应在上游完成后传入 `text`。

## 注意事项

- `model.safetensors` 等大权重只保存在本地，GitHub 不包含它们。
- 重新训练必须传新的 `output_name`，例如 `medbert_v21`，否则会覆盖当前 `medbert_v20` 部署权重。
- 原始预训练 MedBERT 下载完成后放在 `model/pretrained/medbert-base-chinese/`；正式训练推荐使用 AutoDL GPU 环境。
