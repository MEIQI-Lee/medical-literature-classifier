"""Real, local-weight tests. No downloads, training, or dataset writes."""

import io
import os

import pytest
from conftest import make_pdf
from docx import Document

from agent import AcademicClassificationAgent, create_medbert_agent
from preprocess.clean import build_text_input

pytestmark = [
    pytest.mark.real_model,
    pytest.mark.skipif(
        os.environ.get("ACADEMIC_REAL_MODEL_TESTS") != "1",
        reason="Set ACADEMIC_REAL_MODEL_TESTS=1 to load the local 409 MB checkpoint",
    ),
]


@pytest.fixture(scope="module")
def real_agent():
    return create_medbert_agent(device="cpu")


def test_adapter_matches_training_collator_forward(real_agent):
    import torch

    from model.train import _collate

    text = build_text_input("心血管疾病研究", ["高血压"], "分析高血压患者的治疗效果与危险因素。")
    backend = real_agent.classifier.backend
    batch = _collate(backend.loaded.tokenizer, backend.loaded.max_length)([(text, 0)])
    batch.pop("labels")
    with torch.inference_mode():
        expected = torch.softmax(backend.loaded.model(**batch).logits.float(), dim=-1)[0].tolist()
    result = backend.predict(text, top_k=110)
    assert len(result.predictions) == 110
    assert sum(score.score for score in result.predictions) == pytest.approx(1, abs=1e-6)
    for score in result.predictions:
        assert score.score == pytest.approx(expected[score.label_id], abs=1e-7)
    assert result.input_tokens == batch["input_ids"].shape[1]
    assert not result.truncated


def test_real_service_topk_thresholds_and_reused_model(real_agent):
    service = AcademicClassificationAgent(core=real_agent)
    model = real_agent.classifier.backend.loaded.model
    text = "分析高血压患者的心血管疾病风险与治疗效果，比较不同干预措施。"
    normal = service.classify_text(text, top_k=1, confidence_threshold=0)
    low = service.classify_text(text, top_k=1, confidence_threshold=1)
    assert normal.success and low.success
    assert normal.data["returned_top_k"] == 1
    assert low.data["returned_top_k"] == 3 and low.data["low_confidence"]
    assert normal.data["prediction"]["top1"] == low.data["prediction"]["top1"]
    assert real_agent.classifier.backend.loaded.model is model


def test_real_long_input_reports_truncation(real_agent):
    result = real_agent.classify_text("高血压患者的治疗研究与心血管危险因素。" * 100, top_k=3)
    assert result.status == "success"
    assert result.input.truncated and result.input.input_tokens > 512
    assert any(w.code == "MODEL_INPUT_TRUNCATED" for w in result.warnings)


@pytest.mark.parametrize("file_type", ["txt", "pdf", "docx"])
def test_real_uploaded_documents(real_agent, file_type):
    text = (
        "标题：高血压研究\n关键词：高血压；心血管疾病\n摘要：分析高血压患者的危险因素和治疗效果。"
    )
    if file_type == "txt":
        content = text.encode("utf-8")
    elif file_type == "pdf":
        content = make_pdf("Cardiovascular disease and hypertension treatment research.")
    else:
        document = Document()
        for line in text.splitlines():
            document.add_paragraph(line)
        buffer = io.BytesIO()
        document.save(buffer)
        content = buffer.getvalue()
    result = AcademicClassificationAgent(core=real_agent).classify_file(
        content,
        f"paper.{file_type}",
        top_k=3,
    )
    assert result.success, result.to_dict()
    assert result.data["input"]["source_type"] == file_type
    assert len(result.data["prediction"]["topk"]) == 3
    assert any(step.name == "parse_document" for step in result.steps)
