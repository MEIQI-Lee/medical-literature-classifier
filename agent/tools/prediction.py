from pydantic import ValidationError

from agent.adapters import Classifier
from agent.catalog import CategoryCatalog
from agent.errors import AgentError
from agent.schemas import Candidate, ConfidencePolicy, ModelInfo, ModelOutput


def classify_text(
    classifier: Classifier, text: str, *, top_k: int, expected_info: ModelInfo
) -> ModelOutput:
    try:
        output = ModelOutput.model_validate(classifier.predict(text, top_k=top_k))
    except ValidationError as exc:
        raise AgentError("INVALID_MODEL_OUTPUT", "模型输出不符合推理协议。") from exc
    if output.model_info != expected_info:
        raise AgentError(
            "MODEL_VERSION_MISMATCH", "推理输出的模型信息与启动时不一致，请重新加载模型。"
        )
    ids = [prediction.label_id for prediction in output.predictions]
    if len(ids) != min(top_k, len(expected_info.label_ids)):
        raise AgentError("INVALID_MODEL_OUTPUT", "模型返回的候选数量与请求不一致。")
    if len(set(ids)) != len(ids) or not set(ids).issubset(expected_info.label_ids):
        raise AgentError("INVALID_MODEL_OUTPUT", "模型返回了重复或未知的标签。")
    if expected_info.score_type == "probability":
        scores = [prediction.score for prediction in output.predictions]
        if any(score < 0 or score > 1 for score in scores) or sum(scores) > 1 + 1e-6:
            raise AgentError(
                "INVALID_MODEL_OUTPUT", "概率必须在 0 到 1 之间，Top-K 概率之和不能大于 1。"
            )
        if len(scores) == len(expected_info.label_ids) and abs(sum(scores) - 1) > 1e-5:
            raise AgentError("INVALID_MODEL_OUTPUT", "返回全部类别时概率之和必须为 1。")
    return output.model_copy(
        update={
            "predictions": sorted(output.predictions, key=lambda item: (-item.score, item.label_id))
        }
    )


def select_candidates(output: ModelOutput, policy: ConfidencePolicy | None) -> str:
    if policy is None or output.model_info.score_type != "probability":
        return "unknown"
    first, second = output.predictions[:2]
    if first.score < policy.min_probability or first.score - second.score < policy.min_margin:
        return "low"
    return "normal"


def query_categories(output: ModelOutput, catalog: CategoryCatalog) -> list[Candidate]:
    results = []
    for rank, prediction in enumerate(output.predictions, start=1):
        category = catalog.query(prediction.label_id)
        results.append(
            Candidate(
                rank=rank,
                label_id=prediction.label_id,
                category_code=category.code,
                category_name=category.name,
                category_path=category.path,
                score=prediction.score,
                confidence=prediction.score
                if output.model_info.score_type == "probability"
                else None,
            )
        )
    return results
