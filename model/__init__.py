"""model 包初始化。

采用延迟导入，使仅查看 CLI 帮助或运行数据处理时不必预装 torch/transformers。
"""

__all__ = ["train_model", "evaluate_model", "classify", "load_model"]


def __getattr__(name):
    if name == "train_model":
        from model.train import train_model
        return train_model
    if name == "evaluate_model":
        from model.evaluate import evaluate_model
        return evaluate_model
    if name in {"classify", "load_model"}:
        from model.classifier import classify, load_model
        return {"classify": classify, "load_model": load_model}[name]
    raise AttributeError(name)
