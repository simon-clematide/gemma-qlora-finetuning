"""Integrity checks for the portable, recorded classroom replay bundle."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

DEFAULT_CLASSROOM_REPLAY = Path("results/replays/2026-10-18_mps_lora_n1000_seed42/manifest.json")


def load_replay_manifest(manifest_path: str | Path = DEFAULT_CLASSROOM_REPLAY) -> dict[str, Any]:
    """Verify bundled hashes and recorded training settings before using replay files.

    This loads no model, adapter, tokenizer, or ML dependency. Files listed in
    sha256.json are relative to the manifest's directory, so the portable bundle
    can also be verified independently of the full local experiment archive.
    """
    path = Path(manifest_path)
    inventory = json.loads((path.parent / "sha256.json").read_text(encoding="utf-8"))
    required = {
        path.name, "baseline_predictions.jsonl", "finetuned_predictions.jsonl",
        "training_summary.json", "training_config.yaml", "comparison.json",
        "masking_example.json", "source_manifest.json", "source_sha256.json",
        "reload_verification.json", "weights_audit.json",
    }
    if not required <= inventory.keys():
        raise ValueError("Replay hash inventory does not cover all required artifacts")
    for name, expected in inventory.items():
        artifact = path.parent / name
        if not artifact.is_file() or hashlib.sha256(artifact.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Replay artifact hash mismatch: {name}")
    source_inventory = json.loads((path.parent / "source_sha256.json").read_text(encoding="utf-8"))
    for portable_name, source_name in [
        ("baseline_predictions.jsonl", "baseline_predictions.jsonl"),
        ("finetuned_predictions.jsonl", "finetuned_predictions.jsonl"),
        ("training_summary.json", "training_summary.json"),
        ("comparison.json", "comparison.json"),
        ("source_manifest.json", "manifest.json"),
    ]:
        if inventory[portable_name] != source_inventory[source_name]:
            raise ValueError(f"Replay artifact differs from original recording: {portable_name}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("status") != "complete" or manifest.get("execution_mode") != "replay":
        raise ValueError("Replay requires a completed recorded experiment")
    prompt_config = Path("configs/prompts.yaml")
    if prompt_config.is_file() and hashlib.sha256(prompt_config.read_bytes()).hexdigest() != manifest["prompt_config_sha256"]:
        raise ValueError("Replay provenance mismatch: current prompt configuration differs from recording")
    summary = json.loads((path.parent / "training_summary.json").read_text(encoding="utf-8"))
    checks = {
        "model revision": summary["model_revision"] == manifest["model_revision"],
        "base model": summary["base_model_id"] == manifest["base_model"],
        "hardware": summary["device"] == manifest["hardware"]["device"],
        "quantization": summary["loaded_in_4bit"] == (manifest["quantization"] != "none"),
        "step count": summary["actual_training_settings"]["global_step"] == manifest["max_steps"],
        "seed": summary["seed"] == manifest["seed"],
        "training examples": summary["tokenized_training_doc_ids"] == manifest["training_doc_ids"],
        "validation examples": summary["tokenized_validation_doc_ids"] == manifest["validation_doc_ids"],
        "training subset size": len(summary["tokenized_training_doc_ids"]) == manifest["training_subset_size"],
        "batch size": summary["actual_training_settings"]["effective_batch_size"] == manifest["effective_batch_size"],
        "sequence length": summary["actual_training_settings"]["max_length"] == manifest["max_seq_length"],
        "adapter dtype": summary["trainable_parameter_dtypes"] == [f"torch.{manifest['adapter_dtype']}"],
    }
    mismatches = [name for name, matches in checks.items() if not matches]
    if mismatches:
        raise ValueError(f"Replay provenance mismatch: {', '.join(mismatches)}")
    return manifest
