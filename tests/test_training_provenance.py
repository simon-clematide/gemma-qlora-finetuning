"""The pinned model revision must reach both pretrained loaders without ML dependencies."""
import json
import sys
import types
from types import SimpleNamespace

import pytest
import yaml

from germeval_llm import training


@pytest.mark.parametrize("revision", [None, "frozen-model-commit"])
def test_training_pins_tokenizer_and_model_revision(tmp_path, monkeypatch, revision):
    monkeypatch.setattr(training, "_patch_optional_peft_dependency_checks", lambda: None)
    from germeval_llm import inference
    monkeypatch.setattr(inference, "detect_compute_device", lambda: {
        "bf16_supported": False, "cuda_available": False, "device": "cpu",
    })
    calls = []
    tokenizer = SimpleNamespace(pad_token="pad", padding_side="right")

    class StopBeforeWeights(Exception):
        pass

    class AutoTokenizer:
        @staticmethod
        def from_pretrained(model_id, **kwargs):
            calls.append(("tokenizer", model_id, kwargs))
            return tokenizer

    class AutoModel:
        @staticmethod
        def from_pretrained(model_id, **kwargs):
            calls.append(("model", model_id, kwargs))
            raise StopBeforeWeights

    stubs = {
        "torch": {"manual_seed": lambda _: None, "bfloat16": "bf16", "float16": "fp16", "float32": "fp32"},
        "datasets": {"Dataset": SimpleNamespace(from_dict=lambda rows: rows)},
        "peft": {"get_peft_model": None, "prepare_model_for_kbit_training": None},
        "transformers": {"AutoModelForCausalLM": AutoModel, "AutoTokenizer": AutoTokenizer},
        "trl": {"SFTConfig": None, "SFTTrainer": None},
    }
    for name, attributes in stubs.items():
        stub = types.ModuleType(name)
        stub.__dict__.update(attributes)
        monkeypatch.setitem(sys.modules, name, stub)
    config = {"base_model_id": "test-model", "training": {}}
    if revision:
        config["model_revision"] = revision
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))
    data = tmp_path / "train.jsonl"
    data.write_text(json.dumps({
        "doc_id": "train-1", "prompt": [{"role": "user", "content": "Text"}],
        "completion": [{"role": "assistant", "content": '{"labels":[]}'}],
    }) + "\n")
    with pytest.raises(StopBeforeWeights):
        training.run_sft_adapter_training(config_path, data, data, tmp_path / "adapter")
    assert [call[0] for call in calls] == ["tokenizer", "model"]
    assert all(call[2]["revision"] == (revision or "main") for call in calls)


@pytest.mark.parametrize(
    "batch,accumulation,t4,expected",
    [(4, 4, True, (2, 8)), (4, 4, False, (4, 4)),
     (1, 4, True, (1, 4)), (2, 4, True, (2, 4)), (3, 3, True, (1, 9))],
)
def test_t4_batch_fallback_preserves_examples_per_step(batch, accumulation, t4, expected):
    config = {"per_device_train_batch_size": batch, "gradient_accumulation_steps": accumulation}
    actual = training.resolve_training_batch_sizes(config, is_t4_class_gpu=t4)
    assert actual == expected
    assert actual[0] * actual[1] == batch * accumulation


def test_default_qlora_t4_batch_matches_lora():
    qlora = training.load_training_config("configs/qlora.yaml")["training"]
    lora = training.load_training_config("configs/lora.yaml")["training"]
    t4 = training.resolve_training_batch_sizes(qlora, is_t4_class_gpu=True)
    mps = training.resolve_training_batch_sizes(lora, is_t4_class_gpu=False)
    assert t4 == (2, 8)
    assert t4[0] * t4[1] == mps[0] * mps[1] == 16
