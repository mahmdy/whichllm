from types import SimpleNamespace

import pytest

from whichllm import cli


def _model(model_id="example/model"):
    return SimpleNamespace(id=model_id)


def test_transformers_dependencies_are_pinned():
    deps, kind = cli._resolve_model_deps(_model(), None)
    assert kind == "transformers"
    assert deps == [
        "transformers==5.17.0",
        "torch==2.14.0",
        "accelerate==1.15.0",
    ]


def test_gguf_dependencies_are_pinned():
    variant = SimpleNamespace(filename="model.Q4_K_M.gguf", quant_type="Q4_K_M")
    deps, kind = cli._resolve_model_deps(_model(), variant)
    assert kind == "gguf"
    assert deps == [
        "llama-cpp-python==0.3.35",
        "huggingface-hub==1.31.0",
    ]


@pytest.mark.parametrize("quant_type", ["AWQ", "GPTQ"])
def test_deprecated_quantization_backends_are_rejected(quant_type, monkeypatch):
    monkeypatch.setattr(
        "whichllm.engine.quantization.infer_non_gguf_quant_type",
        lambda model_id: quant_type,
    )
    with pytest.raises(ValueError, match="deprecated/stale"):
        cli._resolve_model_deps(_model(), None)


def test_generated_transformers_script_is_safe_by_default():
    source = cli._generate_chat_script(_model(), None, 4096, False)
    assert "trust_remote_code=False" in source
    assert "trust_remote_code=True" not in source


def test_generated_transformers_script_allows_explicit_remote_code_opt_in():
    source = cli._generate_chat_script(
        _model(), None, 4096, False, trust_remote_code=True
    )
    assert "trust_remote_code=True" in source


def test_generated_gguf_script_does_not_use_remote_code_execution():
    variant = SimpleNamespace(filename="model.Q4_K_M.gguf", quant_type="Q4_K_M")
    source = cli._generate_chat_script(_model(), variant, 4096, False)
    assert "trust_remote_code" not in source
