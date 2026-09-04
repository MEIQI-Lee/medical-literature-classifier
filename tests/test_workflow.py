import json
from concurrent.futures import ThreadPoolExecutor

import pytest

from agent import AgentConfig, create_demo_agent
from agent.adapters import PythonModelAdapter
from agent.agent import ClassificationAgent
from agent.demo import DemoClassifier
from agent.errors import AgentError
from agent.schemas import ModelOutput, ModelScore
from agent.tools.text import build_model_input, clean_text


def test_high_confidence_path_and_events(demo_agent, cardiac_text):
    events = []
    result = demo_agent.classify_text(cardiac_text, on_step=events.append)
    assert result.status == "success"
    assert result.mode == "demo"
    assert result.prediction.category_code == "DEMO_01"
    assert result.display_mode == "top1"
    assert result.confidence_status == "normal"
    names = [step.step for step in result.steps]
    assert "parse_document" not in names
    assert "accept_prediction" in names and "review_candidates" not in names
    assert len(events) == 2 * len(result.steps)
    assert [event.status for event in events[:2]] == ["running", "success"]
    assert all(event.request_id == result.request_id for event in events)


def test_low_confidence_returns_candidates(demo_agent):
    result = demo_agent.classify_text("本文介绍一种研究方法及其应用，并对实验结果进行讨论。")
    assert result.status == "success"
    assert result.confidence_status == "low"
    assert result.display_mode == "candidates"
    assert len(result.candidates) == 5
    assert "review_candidates" in [step.step for step in result.steps]


@pytest.mark.parametrize(
    "value,code",
    [
        ("", "EMPTY_TEXT"),
        ("   ", "EMPTY_TEXT"),
        ("<p></p><script>secret</script>", "EMPTY_TEXT"),
        ("高血压", "TEXT_TOO_SHORT"),
        (123, "INVALID_INPUT"),
    ],
)
def test_invalid_text_returns_steps_without_inference(demo_agent, value, code):
    result = demo_agent.classify_text(value)
    assert result.status == "error"
    assert result.error.code == code
    assert result.prediction is None
    assert not result.candidates
    assert "classify_text" not in [step.step for step in result.steps]
    assert result.steps[-1].step == "generate_result"


def test_cleaning_preserves_medical_symbols():
    assert clean_text("<p>CD4&lt;200，IL-6  和  COVID-19</p><script>bad()</script>") == (
        "CD4<200,IL-6 和 COVID-19"
    )
    assert clean_text("CD4<200 且数值 > 10") == "CD4<200 且数值 > 10"


def test_structured_input_reuses_preprocessor(demo_agent):
    from agent.schemas import PaperFields

    fields = PaperFields(
        title="心血管研究", keywords=["高血压", "冠心病"], abstract="分析长期治疗效果。"
    )
    result = demo_agent.classify_paper(**fields.model_dump())
    direct = demo_agent.classifier.predict(build_model_input(fields), top_k=5)
    assert result.input.preparation_strategy == "structured_fields"
    assert result.prediction.score == direct.predictions[0].score


def test_topk_probabilities_are_not_renormalized(cardiac_text):
    agent = create_demo_agent(config=AgentConfig(top_k=3))
    result = agent.classify_text(cardiac_text)
    direct = agent.classifier.predict(cardiac_text, top_k=3)
    assert len(result.candidates) == 3
    assert sum(item.score for item in result.candidates) < 1
    assert [item.score for item in result.candidates] == [item.score for item in direct.predictions]
    assert result.confidence_status == "unknown"


class DecisionClassifier(DemoClassifier):
    def describe(self):
        return super().describe().model_copy(update={"score_type": "decision_score"})

    def predict(self, text, top_k=5):
        return ModelOutput(
            model_info=self.describe(),
            predictions=[ModelScore(label_id=i, score=3 - i) for i in range(top_k)],
        )


def test_decision_scores_are_not_confidence(demo_agent, cardiac_text):
    agent = ClassificationAgent(PythonModelAdapter(DecisionClassifier()), demo_agent.catalog)
    result = agent.classify_text(cardiac_text)
    assert result.status == "success"
    assert result.confidence_status == "unknown"
    assert result.prediction.score == 3
    assert all(item.confidence is None for item in result.candidates)


@pytest.mark.parametrize(
    "change,code",
    [
        ({"label_version": "wrong"}, "LABEL_VERSION_MISMATCH"),
        ({"preprocess_version": "wrong"}, "PREPROCESS_VERSION_MISMATCH"),
        ({"label_ids": [0, 1, 2]}, "LABEL_MAPPING_INVALID"),
    ],
)
def test_startup_detects_incompatible_artifacts(demo_agent, change, code):
    class Incompatible(DemoClassifier):
        def describe(self):
            return super().describe().model_copy(update=change)

    with pytest.raises(AgentError) as caught:
        ClassificationAgent(PythonModelAdapter(Incompatible()), demo_agent.catalog)
    assert caught.value.code == code


@pytest.mark.parametrize("broken", ["nan", "duplicate", "probability", "count", "version"])
def test_invalid_model_output_is_rejected(demo_agent, cardiac_text, broken):
    class Broken(DemoClassifier):
        def predict(self, text, top_k=5):
            result = super().predict(text, top_k).model_dump()
            if broken == "nan":
                result["predictions"][0]["score"] = float("nan")
            elif broken == "duplicate":
                result["predictions"][1]["label_id"] = 0
            elif broken == "probability":
                result["predictions"][0]["score"] = 1.5
            elif broken == "count":
                result["predictions"].pop()
            else:
                result["model_info"]["model_version"] = "other"
            return result

    agent = ClassificationAgent(PythonModelAdapter(Broken()), demo_agent.catalog)
    result = agent.classify_text(cardiac_text)
    assert result.status == "error"
    assert result.error.step == "classify_text"
    assert result.error.code in {"INVALID_MODEL_OUTPUT", "MODEL_VERSION_MISMATCH"}


def test_model_failure_preserves_previous_steps(demo_agent, cardiac_text):
    class Broken(DemoClassifier):
        def predict(self, text, top_k=5):
            raise RuntimeError("internal details")

    result = ClassificationAgent(PythonModelAdapter(Broken()), demo_agent.catalog).classify_text(
        cardiac_text
    )
    assert result.error.code == "MODEL_INFERENCE_FAILED"
    assert result.steps[2].status == "success"
    assert "internal details" not in result.model_dump_json()


def test_callback_failure_does_not_cancel_classification(demo_agent, cardiac_text):
    def fail(step):
        raise RuntimeError("client disconnected")

    result = demo_agent.classify_text(cardiac_text, on_step=fail)
    assert result.status == "success"
    assert sum(note.code == "STEP_CALLBACK_FAILED" for note in result.warnings) == 1


def test_concurrent_requests_keep_separate_traces(tmp_path, cardiac_text):
    path = tmp_path / "trace.jsonl"
    agent = create_demo_agent(trace_path=path)
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(agent.classify_text, [cardiac_text] * 8))
    assert len({item.request_id for item in results}) == 8
    assert all(all(step.request_id == item.request_id for step in item.steps) for item in results)
    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(records) == 8
    assert cardiac_text not in path.read_text()


def test_truncation_metadata_is_forwarded(demo_agent, cardiac_text):
    class Truncated(DemoClassifier):
        def predict(self, text, top_k=5):
            return (
                super()
                .predict(text, top_k)
                .model_copy(update={"truncated": True, "input_tokens": 512})
            )

    result = ClassificationAgent(PythonModelAdapter(Truncated()), demo_agent.catalog).classify_text(
        cardiac_text
    )
    assert result.input.truncated and result.input.input_tokens == 512
    assert "MODEL_INPUT_TRUNCATED" in [note.code for note in result.warnings]
