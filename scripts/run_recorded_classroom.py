"""Run and archive a fresh, offline Task C classroom comparison.

Run from the repository root with the ML environment. No prior predictions or
adapter weights are reused. Each model stage uses a separate process.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
MODEL = "google/gemma-2-2b-it"
SEED = 42


def now():
    return datetime.now(timezone.utc).isoformat()


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def update(run, **values):
    path = run / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest.update(values)
    save(path, manifest)
    return manifest


def prepare(run, max_seq_length=1152):
    import yaml
    from germeval_llm.convert import derive_task_c_example, sample_learning_curve_subset, save_derived_jsonl
    from germeval_llm.inference import detect_compute_device
    from germeval_llm.xml_parser import load_canonical_jsonl, save_canonical_jsonl

    hw = detect_compute_device()
    if hw["device"] != "mps" and not hw["cuda_available"]:
        raise RuntimeError("Accelerator unavailable; refusing an accidental full CPU run")
    cached = Path.home() / ".cache/huggingface/hub/models--google--gemma-2-2b-it"
    revision = (cached / "refs/main").read_text().strip()
    snapshot = cached / "snapshots" / revision
    files = {p.name: sha256(p) for p in sorted(snapshot.iterdir()) if p.is_file()}
    train = load_canonical_jsonl("data/canonical/train.jsonl")
    dev = load_canonical_jsonl("data/canonical/dev.jsonl")[:64]
    evaluation = load_canonical_jsonl("data/canonical/test_syn.jsonl")[:150]
    examples = [derive_task_c_example(doc, apply_typo_patches=True) for doc in train]
    selected = sample_learning_curve_subset(examples, 1000, strategy="stratified", seed=SEED)
    eval_texts = {doc.text for doc in evaluation}
    overlap = [ex.doc_id for ex in selected if ex.text in eval_texts]
    if overlap:
        raise RuntimeError(f"Exact train/evaluation text overlap: {overlap}")
    if {ex.doc_id for ex in selected} & {doc.doc_id for doc in evaluation}:
        raise RuntimeError("Train/evaluation ID overlap")
    save_derived_jsonl(selected, run / "data/train.jsonl")
    save_derived_jsonl([derive_task_c_example(doc) for doc in dev], run / "data/dev.jsonl")
    save_canonical_jsonl(evaluation, run / "data/evaluation.jsonl")
    config = yaml.safe_load(Path("configs/lora.yaml").read_text())
    config["model_revision"] = revision
    config["training"]["max_seq_length"] = max_seq_length
    (run / "training_config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    source_paths = sorted(Path("src/germeval_llm").glob("*.py")) + sorted(Path("configs").glob("*.yaml")) + sorted(Path("tests").glob("test_*.py"))
    source_paths += [Path(p) for p in ["pyproject.toml", "uv.lock", "AGENTS.md", "scripts/run_recorded_classroom.py"]]
    for source in source_paths:
        if source.exists():
            target = run / "source" / source
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
    diff = subprocess.check_output(["git", "diff", "--binary"])
    (run / "source/worktree.patch").write_bytes(diff)
    versions = {}
    for package in ["torch", "transformers", "datasets", "accelerate", "peft", "trl", "huggingface-hub", "numpy", "pydantic", "tokenizers"]:
        versions[package] = importlib.metadata.version(package)
    manifest = {
        "experiment_id": run.name, "execution_mode": "fresh_model_execution", "status": "prepared",
        "created_at_utc": now(), "base_model": MODEL, "model_revision": revision,
        "model_snapshot_sha256": files,
        "code_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "worktree_status": subprocess.check_output(["git", "status", "--short"], text=True),
        "worktree_patch_sha256": hashlib.sha256(diff).hexdigest(),
        "python": sys.version, "platform": platform.platform(), "packages": versions,
        "hardware": hw, "method": "LoRA", "quantization": "none", "seed": SEED,
        "training_split": "train", "validation_split": "dev", "evaluation_split": "test_syn",
        "training_subset_size": 1000, "max_steps": 100, "sampling_strategy": "stratified",
        "training_doc_ids": [ex.doc_id for ex in selected], "validation_doc_ids": [doc.doc_id for doc in dev],
        "evaluation_doc_ids": [doc.doc_id for doc in evaluation], "evaluation_num_docs": len(evaluation),
        "evaluation_selection": "first 150 documents in canonical source order; fixed before training",
        "exact_train_evaluation_text_overlap": overlap,
        "prompt_id": "p1_guidelines", "prompt_config_sha256": sha256("configs/prompts.yaml"),
        "generation": {"max_new_tokens": 128, "do_sample": False, "add_special_tokens": False},
        "gold_typo_patches": False, "training_typo_patches": True,
        "baseline_prediction_file": str(run / "baseline_predictions.jsonl"),
        "finetuned_prediction_file": str(run / "finetuned_predictions.jsonl"),
        "adapter_dir": str(run / "adapter"), "training_log_file": str(run / "training_summary.json"),
        "data_sha256": {str(p.relative_to(run)): sha256(p) for p in sorted((run / "data").iterdir())},
        "offline": True, "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
    }
    save(run / "manifest.json", manifest)
    print(f"Prepared fixed experiment: {run}; hardware={hw['device']}; model revision={revision}", flush=True)


def inference_stage(run, stage):
    import torch
    from germeval_llm.inference import load_gemma_model_and_tokenizer, build_hf_generator_fn, run_task_c_inference_with_generator
    from germeval_llm.xml_parser import load_canonical_jsonl

    manifest = json.loads((run / "manifest.json").read_text())
    torch.manual_seed(SEED)
    adapter = run / "adapter" if stage != "baseline" else None
    if adapter is not None and not (adapter / "adapter_model.safetensors").is_file():
        raise RuntimeError("Missing fresh adapter; refusing base-model fallback")
    model, tokenizer = load_gemma_model_and_tokenizer(
        model_id=MODEL, revision=manifest["model_revision"], adapter_path=adapter, load_in_4bit=False,
    )
    generate = build_hf_generator_fn(model, tokenizer, max_new_tokens=128)
    docs = load_canonical_jsonl(run / "data/evaluation.jsonl")
    if stage == "verify":
        expected = [json.loads(line) for line in (run / "finetuned_predictions.jsonl").read_text().splitlines()]
        checks = []
        for index in [0, 50, 149]:
            actual = run_task_c_inference_with_generator(
                [docs[index]], run_id=run.name + "_verification", split="test_syn", generate_fn=generate,
            )[0]
            reference = expected[index]
            same = actual.raw_output == reference["raw_output"] and actual.parsed_labels == reference["parsed_labels"]
            checks.append({"doc_id": actual.doc_id, "raw_output_equal": same})
            if not same:
                raise RuntimeError(f"Reloaded adapter does not reproduce prediction {index}")
        save(run / "reload_verification.json", {"checks": checks, "all_equal": True})
        return
    with (run / f"{stage}_predictions.jsonl").open("x") as outputs, (run / f"{stage}_prompts.jsonl").open("x") as prompts:
        for i, doc in enumerate(docs):
            captured = {}

            def recorded_generate(prompt):
                captured["formatted_prompt"] = prompt
                captured["prompt_sha256"] = hashlib.sha256(prompt.encode()).hexdigest()
                return generate(prompt)

            pred = run_task_c_inference_with_generator(
                [doc], run_id=run.name + "_" + stage, split="test_syn", generate_fn=recorded_generate,
            )[0]
            outputs.write(pred.model_dump_json() + "\n")
            outputs.flush()
            prompts.write(json.dumps({"doc_id": doc.doc_id, **captured}, ensure_ascii=False) + "\n")
            prompts.flush()
            if (i + 1) % 5 == 0 or i == 0:
                print(f"{now()} {stage}: {i + 1}/{len(docs)} fresh predictions", flush=True)
    save(run / f"{stage}_model.json", {
        "revision": manifest["model_revision"], "device": str(next(model.parameters()).device),
        "parameter_dtypes": dict(Counter(str(p.dtype) for p in model.parameters())),
        "adapter_loaded": adapter is not None, "loaded_in_4bit": bool(getattr(model, "is_loaded_in_4bit", False)),
        "eos_token_id": tokenizer.eos_token_id, "end_of_turn_token_id": tokenizer.convert_tokens_to_ids("<end_of_turn>"),
    })


def evaluate(run):
    from germeval_llm.evaluation import compare_task_c_models, verify_experiment_manifest
    from germeval_llm.inference import load_predictions_jsonl
    from germeval_llm.parsing import parse_task_c_output
    from germeval_llm.reporting import plot_training_and_validation_loss, plot_before_after_metrics_comparison
    from germeval_llm.xml_parser import load_canonical_jsonl
    import matplotlib.pyplot as plt

    manifest = json.loads((run / "manifest.json").read_text())
    docs = load_canonical_jsonl(run / "data/evaluation.jsonl")
    verify_experiment_manifest(manifest, "test_syn", "p1_guidelines", docs, MODEL)
    predictions = [load_predictions_jsonl(run / f"{stage}_predictions.jsonl") for stage in ["baseline", "finetuned"]]
    independent = {}
    for stage, records in zip(["baseline", "finetuned"], predictions):
        tp = fp = fn = 0
        for doc, pred in zip(docs, records):
            assert parse_task_c_output(pred.raw_output) == (pred.parsed_labels, pred.parse_status, pred.parse_errors)
            assert doc.doc_id == pred.doc_id
            gold = {f"{op.category}:{op.polarity}" for op in doc.opinions}
            proposed = set(pred.parsed_labels)
            tp += len(gold & proposed)
            fp += len(proposed - gold)
            fn += len(gold - proposed)
        independent[stage] = {"tp": tp, "fp": fp, "fn": fn, "micro_f1": 2 * tp / (2 * tp + fp + fn)}
    comparison = compare_task_c_models(docs, *predictions, expected_split="test_syn", n_bootstrap=1000, seed=SEED)
    for stage in ["baseline", "finetuned"]:
        assert abs(independent[stage]["micro_f1"] - comparison[stage]["task_c2_category_sentiment_strict"]["micro_f1"]) < 1e-12
    assert (run / "baseline_prompts.jsonl").read_bytes() == (run / "finetuned_prompts.jsonl").read_bytes()
    save(run / "comparison.json", comparison)
    save(run / "independent_scoring.json", independent)
    summary = json.loads((run / "training_summary.json").read_text())
    fig = plot_training_and_validation_loss(summary["train_history"], summary["eval_history"])
    fig.savefig(run / "loss.png", bbox_inches="tight")
    plt.close(fig)
    fig = plot_before_after_metrics_comparison(comparison, method_label="Fresh MPS LoRA" if manifest["hardware"]["device"] == "mps" else "Fresh LoRA")
    fig.savefig(run / "comparison.png", bbox_inches="tight")
    plt.close(fig)
    ci = comparison["paired_bootstrap_c2"]
    lines = ["# Fresh recorded classroom experiment", "", f"Run: {run.name}", "",
             f"Hardware: {manifest['hardware']['device']}; LoRA without quantization.",
             "Training: 1,000 train examples, 100 optimizer steps; validation: first 64 dev documents.",
             "Evaluation: same first 150 test_syn documents; both prediction sets freshly generated.", "",
             "| Metric | Baseline | Adapter |", "| --- | ---: | ---: |"]
    for label, key in [("Exact JSON validity", "valid_exact_rate"), ("Schema validity", "schema_validity_rate")]:
        lines.append(f"| {label} | {comparison['baseline'][key]:.4f} | {comparison['finetuned'][key]:.4f} |")
    lines.append(f"| C2 strict micro-F1 | {independent['baseline']['micro_f1']:.4f} | {independent['finetuned']['micro_f1']:.4f} |")
    lines += ["", f"Paired C2 difference: {ci['point_delta_f1']:+.4f}; 95% document-bootstrap interval [{ci['delta_ci95_low']:+.4f}, {ci['delta_ci95_high']:+.4f}].", "",
              "This is one seed and a fixed demonstration slice. The interval does not measure training-seed variability. This run does not measure CUDA QLoRA."]
    (run / "REPORT.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(independent, indent=2), flush=True)


def stage(run, name):
    update(run, status="running", current_stage=name, stage_started_at_utc=now())
    if name in ["baseline", "finetuned", "verify"]:
        inference_stage(run, name)
    elif name == "train":
        from germeval_llm.training import run_sft_adapter_training
        summary = run_sft_adapter_training(
            run / "training_config.yaml", run / "data/train.jsonl", run / "data/dev.jsonl", run / "adapter",
            max_steps_override=100, subset_size=1000, sampling_strategy="stratified", seed_override=SEED,
        )
        save(run / "training_summary.json", summary)
        if summary["actual_training_settings"]["global_step"] != 100:
            raise RuntimeError("Training did not complete all 100 steps")
        update(run, actual_training=summary["actual_training_settings"], trainable_parameter_dtypes=summary["trainable_parameter_dtypes"])
    elif name == "evaluate":
        evaluate(run)
    completed = json.loads((run / "manifest.json").read_text()).get("completed_stages", [])
    update(run, completed_stages=completed + [name], stage_finished_at_utc=now())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--resume", action="store_true", help="Restart failed training after recording a preparation correction")
    parser.add_argument("--max-seq-length", type=int, default=1152)
    parser.add_argument("--stage", choices=["baseline", "train", "finetuned", "verify", "evaluate"])
    args = parser.parse_args()
    os.chdir(ROOT)
    run = args.run_dir.resolve()
    if args.stage:
        stage(run, args.stage)
        return
    if args.resume:
        import yaml
        previous = json.loads((run / "manifest.json").read_text())
        if previous["status"] != "failed" or previous.get("completed_stages") != ["baseline"]:
            raise RuntimeError("Resume is restricted to a failed first training attempt after a fresh baseline")
        # Keep the failed attempt and the exact source used to generate the fresh baseline.
        attempt = run / "attempts" / f"attempt_{len(list((run / 'attempts').glob('attempt_*'))) + 1:02d}"
        attempt.mkdir(parents=True)
        for name in ["manifest.json", "execution.log", "source", "adapter", "sha256.json", "training_config.yaml"]:
            source = run / name
            if source.exists():
                shutil.move(source, attempt / name)
        config = yaml.safe_load((attempt / "training_config.yaml").read_text())
        config["training"]["max_seq_length"] = args.max_seq_length
        (run / "training_config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
        for source in sorted(Path("src/germeval_llm").glob("*.py")) + [Path(__file__).relative_to(ROOT)]:
            destination = run / "source" / source
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
        (run / "source/worktree.patch").write_bytes(subprocess.check_output(["git", "diff", "--binary"]))
        previous.update(
            status="prepared", current_stage=None, error=None,
            preparation_correction={
                "reason": "1024-token limit dropped 10 examples and truncated other completions; stopped before any completed optimizer step",
                "new_max_seq_length": args.max_seq_length, "observed_max_training_sequence_tokens": 1134,
                "archived_attempt": str(attempt), "baseline_source_snapshot": str(attempt / "source"),
                "corrected_training_source_snapshot": str(run / "source"),
                "no_hyperparameter_selection_from_scores": True,
            },
        )
        save(run / "manifest.json", previous)
    else:
        run.mkdir(parents=True, exist_ok=False)
    try:
        if not args.resume:
            prepare(run, args.max_seq_length)
        environment = dict(os.environ, PYTHONHASHSEED=str(SEED), HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false")
        with (run / "execution.log").open("w") as log:
            stages = ["train", "finetuned", "verify", "evaluate"] if args.resume else ["baseline", "train", "finetuned", "verify", "evaluate"]
            for name in stages:
                print(f"{now()} starting {name}", flush=True)
                process = subprocess.Popen(
                    [sys.executable, "-u", str(Path(__file__).resolve()), "--run-dir", str(run), "--stage", name],
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=environment,
                )
                for line in process.stdout:
                    print(line, end="", flush=True)
                    log.write(line)
                    log.flush()
                if process.wait() != 0:
                    raise RuntimeError(f"Stage {name} failed; see execution.log")
        update(run, status="complete", current_stage=None, completed_at_utc=now())
    except BaseException:
        if (run / "manifest.json").exists():
            update(run, status="failed", failure_at_utc=now(), error=traceback.format_exc())
        raise
    finally:
        save(run / "sha256.json", {
            str(path.relative_to(run)): sha256(path)
            for path in sorted(run.rglob("*")) if path.is_file() and path != run / "sha256.json"
        })


if __name__ == "__main__":
    main()
