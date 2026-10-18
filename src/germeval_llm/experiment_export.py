"""Portable, weight-free exports of completed live classroom experiments."""
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from germeval_llm.evaluation import align_predictions, compare_task_c_models
from germeval_llm.inference import save_predictions_jsonl
from germeval_llm.schemas import CanonicalDocument, PredictionRecord


def export_live_experiment(
    *,
    output_root: str | Path,
    eval_docs: Sequence[CanonicalDocument],
    baseline_predictions: Sequence[PredictionRecord],
    finetuned_predictions: Sequence[PredictionRecord],
    training_summary: dict[str, Any],
    training_config: str | Path,
    prompt_config: str | Path = "configs/prompts.yaml",
    dataset_config: str | Path = "configs/dataset.yaml",
    gold_typo_patches: bool = False,
    sampling_strategy: str = "stratified",
    inference_batch_size: int = 1,
) -> Path:
    """Validate paired predictions and export a new live CUDA run as a ZIP.

    This format is independent of the historical classroom replay format: it
    makes no claims about source archives, weight audits or reload equivalence.
    """
    summary = training_summary
    if summary["device"] != "cuda" or summary["hardware"]["device"] != "cuda":
        raise ValueError("Live export requires a measured CUDA training summary")
    if not eval_docs or any(d.source_split != "test_syn" for d in eval_docs):
        raise ValueError("Live demo export requires nonempty test_syn evaluation documents")
    baseline = align_predictions(eval_docs, baseline_predictions, expected_split="test_syn")
    finetuned = align_predictions(eval_docs, finetuned_predictions, expected_split="test_syn")
    comparison = compare_task_c_models(
        eval_docs, baseline, finetuned, expected_split="test_syn",
        apply_gold_typo_patches=gold_typo_patches, n_bootstrap=1000, seed=summary["seed"],
    )
    now = datetime.now(UTC)
    method = "qlora" if summary["loaded_in_4bit"] else "lora"
    run_id = f"{now:%Y-%m-%d_%H%M%S}_colab_{method}_seed{summary['seed']}_{uuid4().hex[:8]}"
    settings = summary["actual_training_settings"]
    manifest = {
        "format_version": 1,
        "experiment_id": run_id,
        "experiment_type": "live_classroom_experiment",
        "execution_mode": "live",
        "status": "complete",
        "exported_at_utc": now.isoformat(),
        "base_model": summary["base_model_id"],
        "model_revision": summary["model_revision"],
        "method": "QLoRA" if summary["loaded_in_4bit"] else "LoRA",
        "quantization": "nf4_4bit" if summary["loaded_in_4bit"] else "none",
        "adapter_dtypes": summary["trainable_parameter_dtypes"],
        "hardware": summary["hardware"],
        "seed": summary["seed"],
        "training_split": "train", "validation_split": "dev", "evaluation_split": "test_syn",
        "training_doc_ids": summary["tokenized_training_doc_ids"],
        "validation_doc_ids": summary["tokenized_validation_doc_ids"],
        "evaluation_doc_ids": [d.doc_id for d in eval_docs],
        "evaluation_num_docs": len(eval_docs),
        "training_subset_size": len(summary["tokenized_training_doc_ids"]),
        "sampling_strategy": sampling_strategy,
        "max_steps": settings["global_step"],
        "max_seq_length": settings["max_length"],
        "effective_batch_size": settings["effective_batch_size"],
        "prompt_id": "p1_guidelines", "n_shots": 0,
        "generation": {"max_new_tokens": 128, "do_sample": False, "add_special_tokens": False,
                       "batch_size": inference_batch_size,
                       "latency_semantics": "amortized batch wall time per document" if inference_batch_size > 1 else "per-document wall time"},
        "gold_typo_patches": gold_typo_patches,
        "evaluation_selection": "first documents in canonical source order, fixed before training",
        "baseline_prediction_file": "baseline_predictions.jsonl",
        "finetuned_prediction_file": "finetuned_predictions.jsonl",
        "training_log_file": "training_summary.json",
        "training_config_file": "training_config.yaml",
        "artifact_paths_relative_to": "manifest directory",
        "weights_included": False,
    }
    revision = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False)
    manifest["source_commit"] = revision.stdout.strip() if revision.returncode == 0 else None
    dirty = subprocess.run(["git", "diff", "--quiet", "HEAD"], capture_output=True, check=False)
    manifest["source_tracked_changes"] = dirty.returncode != 0
    bundle = Path(output_root) / run_id
    bundle.mkdir(parents=True, exist_ok=False)

    def write_json(name: str, value: Any) -> None:
        (bundle / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    save_predictions_jsonl(baseline, bundle / "baseline_predictions.jsonl")
    save_predictions_jsonl(finetuned, bundle / "finetuned_predictions.jsonl")
    write_json("training_summary.json", summary)
    write_json("comparison.json", comparison)
    # Preserve exact evaluated gold and text so scores can be checked independently.
    (bundle / "evaluation_documents.jsonl").write_text(
        "".join(doc.model_dump_json() + "\n" for doc in eval_docs), encoding="utf-8",
    )
    for source, name in [(training_config, "training_config.yaml"),
                         (prompt_config, "prompts.yaml"), (dataset_config, "dataset.yaml")]:
        shutil.copyfile(source, bundle / name)
    manifest["prompt_config_sha256"] = hashlib.sha256((bundle / "prompts.yaml").read_bytes()).hexdigest()
    write_json("manifest.json", manifest)

    import matplotlib.pyplot as plt

    from germeval_llm.reporting import (
        plot_before_after_metrics_comparison,
        plot_training_and_validation_loss,
    )
    from germeval_llm.training import extract_training_and_eval_series

    history, validation = extract_training_and_eval_series(summary.get("log_history", []))
    figures = {
        "loss.png": plot_training_and_validation_loss(history, validation),
        "comparison.png": plot_before_after_metrics_comparison(
            comparison, method_label=f"Live {manifest['method']}", include_raw_json=False,
        ),
    }
    for name, figure in figures.items():
        figure.savefig(bundle / name, dpi=150, bbox_inches="tight")
        plt.close(figure)
    (bundle / "README.md").write_text(
        "# Live classroom experiment\n\n"
        "Measured live CUDA run; weights are excluded. Paths in manifest.json are relative to this folder.\n"
        "Extract this folder under results/replays/ in the repository, review, then commit it.\n"
        "This live-export format is not the historical load_replay_manifest() format.\n"
        "Verify checksums with verify_export_bundle(path); comparison.json contains paired scores and bootstrap intervals.\n"
        "The summary includes selected training/dev IDs and actual hardware/settings; evaluation_documents.jsonl preserves gold.\n",
        encoding="utf-8",
    )
    write_json("sha256.json", {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(bundle.iterdir()) if p.is_file()
    })
    verify_export_bundle(bundle)
    return Path(shutil.make_archive(str(bundle), "zip", root_dir=bundle.parent, base_dir=bundle.name))


def verify_export_bundle(bundle: str | Path) -> dict[str, Any]:
    """Check the compact live export's inventory before importing or committing."""
    root = Path(bundle)
    inventory = json.loads((root / "sha256.json").read_text(encoding="utf-8"))
    required = {"manifest.json", "baseline_predictions.jsonl", "finetuned_predictions.jsonl",
                "training_summary.json", "training_config.yaml", "comparison.json", "prompts.yaml",
                "dataset.yaml", "evaluation_documents.jsonl", "loss.png", "comparison.png", "README.md"}
    if set(inventory) != required:
        raise ValueError("Export inventory does not match required artifacts")
    for name, expected in inventory.items():
        path = root / name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Export artifact hash mismatch: {name}")
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("execution_mode") != "live" or manifest.get("status") != "complete":
        raise ValueError("Expected a completed live export")
    return manifest
