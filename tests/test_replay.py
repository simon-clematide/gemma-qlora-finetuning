"""Portable replay checks must catch edited artifacts and inconsistent provenance."""
import hashlib
import json
from pathlib import Path
import shutil

import pytest

from germeval_llm.replay import DEFAULT_CLASSROOM_REPLAY, load_replay_manifest


def test_replay_rejects_modified_predictions(tmp_path):
    bundle = tmp_path / "replay"
    shutil.copytree(DEFAULT_CLASSROOM_REPLAY.parent, bundle)
    (bundle / "baseline_predictions.jsonl").write_text("altered\n")
    with pytest.raises(ValueError, match="hash mismatch: baseline_predictions"):
        load_replay_manifest(bundle / "manifest.json")


def test_replay_rejects_inconsistent_training_revision(tmp_path):
    bundle = tmp_path / "replay"
    shutil.copytree(DEFAULT_CLASSROOM_REPLAY.parent, bundle)
    summary_path = bundle / "manifest.json"
    summary = json.loads(summary_path.read_text())
    summary["model_revision"] = "different-model-revision"
    summary_path.write_text(json.dumps(summary))
    # Even if an editor regenerates hashes, semantic provenance must agree.
    index_path = bundle / "sha256.json"
    index = json.loads(index_path.read_text())
    index[summary_path.name] = hashlib.sha256(summary_path.read_bytes()).hexdigest()
    index_path.write_text(json.dumps(index))
    with pytest.raises(ValueError, match="provenance mismatch: model revision"):
        load_replay_manifest(bundle / "manifest.json")


def test_recorded_masking_example_is_a_selected_training_document():
    manifest = load_replay_manifest()
    data = json.loads(Path(manifest["masking_example_file"]).read_text())
    example, diagnostic = data["example"], data["mask_diagnostics"]
    assert example["doc_id"] == diagnostic["doc_id"] == manifest["masking_example_doc_id"]
    assert example["doc_id"] in manifest["training_doc_ids"]
    assert example["source_split"] == "train"
    from germeval_llm.schemas import TaskCExample
    from germeval_llm.prompts import format_gemma_chat_string
    parsed = TaskCExample.model_validate(example)
    assert data["formatted_training_sequence"] == format_gemma_chat_string(parsed.prompt, parsed.completion) + "<eos>"
    masked = diagnostic["masked_prompt_tokens"]
    assert diagnostic["labels"][:masked] == [-100] * masked
    assert diagnostic["labels"][masked:] == diagnostic["input_ids"][masked:]
    assert diagnostic["active_completion_tokens"] > 0
    assert diagnostic["total_tokens"] <= manifest["max_seq_length"]


def test_replay_requires_hash_coverage_for_predictions(tmp_path):
    bundle = tmp_path / "replay"
    shutil.copytree(DEFAULT_CLASSROOM_REPLAY.parent, bundle)
    index_path = bundle / "sha256.json"
    index = json.loads(index_path.read_text())
    del index["baseline_predictions.jsonl"]
    index_path.write_text(json.dumps(index))
    with pytest.raises(ValueError, match="does not cover all required artifacts"):
        load_replay_manifest(bundle / "manifest.json")
