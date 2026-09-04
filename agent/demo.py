import math

from agent.schemas import ModelInfo, ModelOutput, ModelScore


class DemoClassifier:
    """Deterministic keyword fixture for integration, not a trained medical classifier."""

    KEYWORDS = (
        ("心血管", "高血压", "冠心病", "心脏", "cardiovascular", "hypertension"),
        ("肿瘤", "癌症", "化疗", "肺癌", "cancer", "tumor"),
        ("感染", "细菌", "病毒", "抗生素", "infection", "virus"),
        ("神经", "脑卒中", "癫痫", "认知", "neurology", "stroke"),
        ("消化", "胃炎", "肠道", "肝脏", "gastric", "digestive"),
    )

    def describe(self) -> ModelInfo:
        return ModelInfo(
            model_version="demo-keywords-v1",
            label_version="demo-labels-v1",
            preprocess_version="builtin-v1",
            label_ids=list(range(5)),
            score_type="probability",
            calibrated=False,
            mode="demo",
        )

    def predict(self, text: str, top_k: int = 5) -> ModelOutput:
        text = text.casefold()
        logits = [2.4 * sum(word in text for word in words) for words in self.KEYWORDS]
        exponentials = [math.exp(value - max(logits)) for value in logits]
        total = sum(exponentials)
        scores = [
            ModelScore(label_id=i, score=value / total) for i, value in enumerate(exponentials)
        ]
        return ModelOutput(
            model_info=self.describe(),
            predictions=sorted(scores, key=lambda item: (-item.score, item.label_id))[:top_k],
        )
