import json
import sys
from pathlib import Path
from types import ModuleType

import pytest

from agent.adapters import ScriptModelAdapter, run_json_command
from agent.agent import ClassificationAgent
from agent.cli import main
from agent.demo import DemoClassifier
from agent.errors import AgentError
from agent.factory import create_agent
from agent.pipeline import PipelineSettings, run_pipeline
from agent.tools.text import DefaultTextPreprocessor

ROOT = Path(__file__).resolve().parents[1]


def test_invalid_preprocessor_fails_at_startup(demo_agent):
    with pytest.raises(AgentError) as caught:
        ClassificationAgent(demo_agent.classifier, demo_agent.catalog, preprocessor=object())
    assert caught.value.code == "CONFIG_ERROR"


def test_python_factory_loads_once_and_injects_shared_preprocessor(
    tmp_path, monkeypatch, cardiac_text
):
    module = ModuleType("team_integration_fixture")
    calls = {"load": 0, "prepare": 0}
    checkpoint = tmp_path / "fixture.bin"
    checkpoint.write_bytes(b"integration fixture")

    def create_classifier(checkpoint_path):
        assert Path(checkpoint_path) == checkpoint
        calls["load"] += 1
        return DemoClassifier()

    class SharedPreprocessor(DefaultTextPreprocessor):
        def prepare(self, document):
            calls["prepare"] += 1
            return super().prepare(document)

    module.create_classifier = create_classifier
    module.create_preprocessor = SharedPreprocessor
    monkeypatch.setitem(sys.modules, module.__name__, module)
    config = {
        "labels_path": str(ROOT / "agent/data/demo_labels.json"),
        "categories_path": str(ROOT / "agent/data/demo_categories.json"),
        "model": {
            "kind": "python",
            "checkpoint_path": "fixture.bin",
            "factory": "team_integration_fixture:create_classifier",
        },
        "preprocessor_factory": "team_integration_fixture:create_preprocessor",
    }
    path = tmp_path / "agent.json"
    path.write_text(json.dumps(config))
    agent = create_agent(path)
    assert agent.classify_text(cardiac_text).status == "success"
    assert agent.classify_text(cardiac_text).status == "success"
    assert calls == {"load": 1, "prepare": 2}


def test_script_model_matches_python_model(tmp_path, demo_agent, cardiac_text):
    checkpoint = tmp_path / "demo.marker"
    checkpoint.write_text("protocol test only")
    script = ScriptModelAdapter(
        [sys.executable, str(ROOT / "examples/model_script_demo.py")],
        checkpoint,
        cwd=tmp_path,
    )
    agent = ClassificationAgent(script, demo_agent.catalog)
    result = agent.classify_text(cardiac_text)
    expected = demo_agent.classify_text(cardiac_text)
    assert result.status == "success" and result.mode == "demo"
    assert result.candidates == expected.candidates


def test_script_timeout_and_invalid_stdout(tmp_path):
    with pytest.raises(AgentError) as timeout:
        run_json_command(
            [sys.executable, "-c", "import time; time.sleep(5)"], {}, cwd=tmp_path, timeout=0.1
        )
    assert timeout.value.code == "TOOL_TIMEOUT"
    with pytest.raises(AgentError) as output:
        run_json_command(
            [sys.executable, "-c", "print('debug output')"], {}, cwd=tmp_path, timeout=5
        )
    assert output.value.code == "INVALID_TOOL_OUTPUT"


def test_script_nonzero_and_no_shell_expansion(tmp_path):
    with pytest.raises(AgentError) as failure:
        run_json_command([sys.executable, "-c", "raise SystemExit(3)"], {}, cwd=tmp_path, timeout=5)
    assert failure.value.code == "TOOL_EXECUTION_FAILED"
    payload = {"text": "$(touch unexpected); `touch unexpected`; 文本"}
    output = run_json_command(
        [sys.executable, "-c", "import json,sys; print(json.dumps(json.load(sys.stdin)))"],
        payload,
        cwd=tmp_path,
        timeout=5,
    )
    assert output == payload
    assert not (tmp_path / "unexpected").exists()


def test_online_missing_weights_never_imports_training(tmp_path):
    config = {
        "labels_path": "missing-labels.json",
        "categories_path": "missing-categories.json",
        "model": {
            "kind": "python",
            "factory": "must_not_import:train",
            "checkpoint_path": "missing",
        },
    }
    path = tmp_path / "agent.json"
    path.write_text(json.dumps(config))
    with pytest.raises(AgentError) as caught:
        create_agent(path)
    assert caught.value.code == "MODEL_NOT_READY"


def test_factory_resolves_paths_from_config_directory(tmp_path, cardiac_text):
    (tmp_path / "demo.marker").write_text("protocol test only")
    config = {
        "labels_path": str(ROOT / "agent/data/demo_labels.json"),
        "categories_path": str(ROOT / "agent/data/demo_categories.json"),
        "model": {
            "kind": "script",
            "checkpoint_path": "demo.marker",
            "cwd": ".",
            "command": ["{python}", str(ROOT / "examples/model_script_demo.py")],
        },
    }
    path = tmp_path / "agent.json"
    path.write_text(json.dumps(config))
    assert create_agent(path).classify_text(cardiac_text).status == "success"


def test_cli_json_and_exit_codes(capsys, cardiac_text):
    assert main(["--demo", "--text", cardiac_text, "--steps"]) == 0
    captured = capsys.readouterr()
    result = json.loads(captured.out)
    assert result["status"] == "success"
    assert all(
        json.loads(line)["request_id"] == result["request_id"] for line in captured.err.splitlines()
    )
    assert main(["--demo", "--text", ""]) == 1
    assert json.loads(capsys.readouterr().out)["error"]["code"] == "EMPTY_TEXT"


def test_offline_default_skips_training_with_weights(tmp_path):
    (tmp_path / "weights.bin").write_bytes(b"fixture")
    settings = PipelineSettings(checkpoint_path="weights.bin")
    result = run_pipeline(settings, base=tmp_path, train_if_missing=True)
    assert result["status"] == "success"
    assert all(item["status"] == "skipped" for item in result["steps"])


def test_offline_default_fails_without_weights(tmp_path):
    result = run_pipeline(PipelineSettings(checkpoint_path="missing"), base=tmp_path)
    assert result["error"]["code"] == "MODEL_NOT_READY"


def stage_settings(tmp_path, name, output):
    script = tmp_path / f"{name}.py"
    script.write_text(
        "import json, pathlib, sys\n"
        "payload = json.load(sys.stdin)\n"
        f"pathlib.Path({output!r}).write_text(payload['action'])\n"
        "print(json.dumps({'status': 'success'}))\n"
    )
    return {"command": ["{python}", str(script)], "outputs": [output]}


def test_offline_explicit_prepare_and_training(tmp_path):
    settings = PipelineSettings.model_validate(
        {
            "checkpoint_path": "weights.bin",
            "stages": {
                "collect": stage_settings(tmp_path, "collect", "raw.jsonl"),
                "clean": stage_settings(tmp_path, "clean", "train.jsonl"),
                "train": stage_settings(tmp_path, "train", "weights.bin"),
            },
        }
    )
    result = run_pipeline(settings, base=tmp_path, prepare_data=True, train_if_missing=True)
    assert result["status"] == "success"
    assert [step["step"] for step in result["steps"]] == ["collect", "clean", "train"]
    assert (tmp_path / "weights.bin").read_text() == "train"


def test_offline_missing_artifact_stops_later_stages(tmp_path):
    settings = PipelineSettings.model_validate(
        {
            "checkpoint_path": "weights.bin",
            "stages": {
                "collect": {
                    "command": ["{python}", "-c", 'print(\'{"status": "success"}\')'],
                    "outputs": ["missing.jsonl"],
                },
                "clean": stage_settings(tmp_path, "clean", "train.jsonl"),
                "train": stage_settings(tmp_path, "train", "weights.bin"),
            },
        }
    )
    result = run_pipeline(settings, base=tmp_path, prepare_data=True, train_if_missing=True)
    assert result["error"]["code"] == "MISSING_ARTIFACT"
    assert not (tmp_path / "train.jsonl").exists()
    assert not (tmp_path / "weights.bin").exists()
