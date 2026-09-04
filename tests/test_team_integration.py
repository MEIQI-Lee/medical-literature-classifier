import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import ModuleType

import pytest

from agent import AcademicClassificationAgent
from agent.errors import AgentError
from agent.integrations.medbert_backend import catalog_from_bundle, inspect_bundle
from agent.integrations.team_preprocessor import TeamPreprocessor
from agent.schemas import PaperFields, ParsedDocument
from agent.tools import get_all_tool_specs, get_tool
from agent.tools.base import ToolResult
from preprocess.clean import build_text_input

ROOT = Path(__file__).resolve().parents[1]


def test_preprocessor_matches_training_and_preserves_serialized_fields():
    fields = PaperFields(title="糖尿病研究", keywords=["胰岛素", "血糖"], abstract="分析治疗效果。")
    expected = build_text_input(**fields.model_dump())
    prepare = TeamPreprocessor().prepare
    assert prepare(ParsedDocument(text="", source_type="paper", fields=fields)).text == expected
    assert prepare(ParsedDocument(text=expected, source_type="text")).text == expected
    assert "标题：" in expected


def test_preprocessor_document_headings_precede_whitespace_cleaning():
    text = (
        "标题：临床研究\n关键词：糖尿病；血糖\n摘要：研究药物治疗效果。\n1 引言\n不属于摘要的正文"
    )
    result = TeamPreprocessor().prepare(ParsedDocument(text=text, source_type="pdf"))
    assert result.strategy == "document_fields"
    assert result.text == build_text_input("临床研究", ["糖尿病", "血糖"], "研究药物治疗效果。")


def test_preprocessor_free_text_preserves_comparisons():
    result = TeamPreprocessor().prepare(
        ParsedDocument(
            text="<p>研究 CD4&lt;200 且 P&gt;0.05 的患者</p>",
            source_type="text",
        )
    )
    assert "CD4<200" in result.text and "P>0.05" in result.text
    assert result.strategy == "free_text"


def test_service_contract_and_live_steps(demo_agent, cardiac_text):
    service = AcademicClassificationAgent(core=demo_agent)
    events = []
    result = service.classify_text(
        cardiac_text, top_k=1, confidence_threshold=0, on_step=events.append
    )
    payload = result.to_dict()
    assert isinstance(result, ToolResult) and result.success
    assert set(payload) == {"success", "data", "error", "steps", "logs"}
    assert set(payload["data"]["prediction"]["top1"]) == {"code", "name", "confidence"}
    assert payload["data"]["returned_top_k"] == 1
    assert len(events) == 2 * len(payload["steps"])
    assert events[0]["status"] == "running" and events[1]["status"] == "done"
    assert [e["step_id"] for e in events[1::2]] == [s["step_id"] for s in payload["steps"]]


def test_low_confidence_expands_top1_request(demo_agent, cardiac_text):
    result = demo_agent.classify_text(cardiac_text, top_k=1, confidence_threshold=1)
    assert result.status == "success" and result.confidence_status == "low"
    assert result.requested_top_k == 1 and result.returned_top_k == 3


def test_topk_is_capped_by_available_labels(demo_agent, cardiac_text):
    result = demo_agent.classify_text(cardiac_text, top_k=500)
    assert result.status == "success" and len(result.candidates) == 5
    assert result.requested_top_k == 500


@pytest.mark.parametrize(
    "options",
    [
        {"top_k": 0},
        {"top_k": True},
        {"top_k": "3"},
        {"confidence_threshold": float("nan")},
        {"confidence_threshold": float("inf")},
        {"confidence_threshold": True},
        {"confidence_threshold": -0.1},
    ],
)
def test_bad_request_options_fail_before_inference(demo_agent, cardiac_text, options):
    result = demo_agent.classify_text(cardiac_text, **options)
    assert result.error.code == "INVALID_INPUT"
    assert "classify_text" not in [step.step for step in result.steps]


def test_concurrent_options_do_not_mutate_agent(demo_agent, cardiac_text):
    original = demo_agent.config.model_dump()
    service = AcademicClassificationAgent(core=demo_agent)

    def classify(index):
        return service.classify_text(
            cardiac_text, top_k=1 if index % 2 else 4, confidence_threshold=0 if index % 2 else 1
        )

    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(classify, range(8)))
    assert all(result.success for result in results)
    assert [r.data["returned_top_k"] for r in results] == [4, 1] * 4
    assert len({r.data["request_id"] for r in results}) == 8
    assert demo_agent.config.model_dump() == original


def test_lazy_factory_is_called_once_under_concurrency(demo_agent, monkeypatch):
    calls = []

    def factory(**kwargs):
        calls.append(kwargs)
        return demo_agent

    monkeypatch.setattr("agent.factory.create_medbert_agent", factory)
    service = AcademicClassificationAgent()
    assert calls == []
    with ThreadPoolExecutor(max_workers=4) as executor:
        list(executor.map(lambda _: service.warmup(), range(8)))
    assert len(calls) == 1


def test_existing_import_paths_and_registry():
    from agent import get_agent
    from agent.agent import get_agent as legacy_get_agent

    assert legacy_get_agent is get_agent
    assert {tool["name"] for tool in get_all_tool_specs()} >= {
        "clean_dataset",
        "train_model",
        "classify_text",
        "collect_data",
        "query_category",
    }


def test_task_routing_does_not_treat_article_content_as_training(demo_agent):
    service = AcademicClassificationAgent(core=demo_agent)
    text = "训练数据包括高血压患者，用于分析心血管风险与治疗效果。"
    assert service.plan(text) == [{"tool": "classify_text", "args": {"text": text}}]
    assert service.plan("训练模型")[0]["args"]["model_type"] == "medbert"
    assert not service.run({"task": []}).success
    assert not service.run({"task": "unknown"}).success


def test_string_keywords_and_invalid_keywords(demo_agent):
    service = AcademicClassificationAgent(core=demo_agent)
    assert service.classify_paper(title="心血管疾病研究", keywords="高血压；冠心病").success
    assert not service.classify_paper(title="心血管疾病研究", keywords=[123]).success


def test_collect_bridge_translates_total_limit_without_real_crawl(monkeypatch):
    stub = ModuleType("crawler.cnki")
    calls = []
    stub.run_full_crawl = lambda **kwargs: (
        calls.append(kwargs)
        or {
            "total_collected": 7,
            "failed_categories": [],
        }
    )
    monkeypatch.setitem(sys.modules, "crawler.cnki", stub)
    result = get_tool("collect_data").run(categories=["R51", "R52"], target_count=7)
    assert result.success
    assert calls[0]["max_per_category"] == 4
    assert result.data["effective_target_limit"] == 8
    assert result.data["collection"]["total_collected"] == 7
    assert not get_tool("collect_data").run(categories=[]).success
    assert len(calls) == 1


def test_query_distinguishes_catalogue_from_model_labels(tmp_path, monkeypatch):
    checkpoint = tmp_path / "medbert_v20/best"
    checkpoint.mkdir(parents=True)
    (checkpoint / "labels.json").write_text((ROOT / "data/labels.json").read_text())
    monkeypatch.setattr("config.CHECKPOINT_DIR", tmp_path)
    supported = get_tool("query_category").run(code="R54")
    excluded = get_tool("query_category").run(code="R-01")
    assert supported.success and supported.data["supported"]
    assert excluded.success and not excluded.data["supported"]


def test_missing_bundle_rejects_marker_without_loading_or_training(tmp_path):
    (tmp_path / ".gitkeep").write_text("placeholder")
    with pytest.raises(AgentError, match="model.safetensors"):
        inspect_bundle(tmp_path)


def test_dataset_fingerprint_is_informational(tmp_path):
    # Metadata-only fixture: this test neither requires nor pretends to load real weights.
    labels = json.loads((ROOT / "data/labels.json").read_text())
    labels["dataset_sha256"] = "a-different-training-dataset"
    (tmp_path / "labels.json").write_text(json.dumps(labels))
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "label2id": labels["label2id"],
                "id2label": labels["id2label"],
                "max_position_embeddings": 512,
            }
        )
    )
    (tmp_path / "training_config.json").write_text(
        json.dumps(
            {
                "model_kind": "sequence_classification",
                "local_files_only": True,
                "max_length": 512,
            }
        )
    )
    for name in [
        "model.safetensors",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.txt",
        "special_tokens_map.json",
    ]:
        (tmp_path / name).write_text("metadata fixture only; not a runnable model")
    bundle = inspect_bundle(tmp_path)
    assert bundle["label_count"] == 110
    assert bundle["dataset_fingerprint_match"] is False
    assert bundle["labels"]["label2id"] == labels["label2id"]
    assert catalog_from_bundle(bundle).query(labels["label2id"]["R54"]).code == "R54"


def test_offline_default_only_checks_existing_assets(monkeypatch, capsys):
    from agent.offline import main

    monkeypatch.setattr("agent.offline.get_tool", lambda name: pytest.fail("unexpected tool"))
    monkeypatch.setattr("agent.offline.inspect_bundle", lambda: {"label_count": 110})
    assert main([]) == 0
    assert json.loads(capsys.readouterr().out)["data"]["training_skipped"] is True
