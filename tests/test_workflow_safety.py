"""Regression coverage for scoring boundaries, CLI demonstrations, and simulations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from germeval_llm.convert import derive_task_d_example, select_few_shot_examples_task_d
from germeval_llm.evaluation import evaluate_task_c_predictions, evaluate_task_d_predictions
from germeval_llm.official_evaluation import export_task_c_to_official_tsv, export_task_d_to_official_xml
from germeval_llm.schemas import CanonicalDocument, CanonicalOpinion, PredictionRecord


def make_doc(split="dev", doc_id="doc", ordinal=0):
    return CanonicalDocument(
        doc_id=doc_id, source_split=split, source_ordinal=ordinal,
        text="Der Zug hat heute wieder eine große Verspätung.",
        relevance=True, document_sentiment="negative",
        opinions=[CanonicalOpinion(
            category_full="Zugfahrt#Haupt", category="Zugfahrt", subcategory="Haupt",
            polarity="negative", target_xml="Zug", start_xml=4, end_xml=7,
            is_implicit=False, is_discontinuous_star=False,
        )],
    )


@pytest.mark.parametrize("task", ["C", "D"])
@pytest.mark.parametrize("problem", ["missing", "duplicate", "unexpected", "split"])
def test_scoring_and_exports_reject_mismatched_predictions(tmp_path, task, problem):
    doc = make_doc()
    pred = PredictionRecord(
        run_id="r", task=task, split="dev", doc_id="doc", source_ordinal=0,
        raw_output="{}", parse_status="VALID_EXACT",
    )
    predictions = {
        "missing": [],
        "duplicate": [pred, pred],
        "unexpected": [pred, pred.model_copy(update={"doc_id": "extra"})],
        "split": [pred.model_copy(update={"split": "test_syn"})],
    }[problem]
    scorer = evaluate_task_c_predictions if task == "C" else evaluate_task_d_predictions
    exporter = export_task_c_to_official_tsv if task == "C" else export_task_d_to_official_xml
    for action in [lambda: scorer([doc], predictions), lambda: exporter([doc], predictions, tmp_path / "out")]:
        with pytest.raises(ValueError):
            action()
    assert not (tmp_path / "out").exists()


def test_task_d_demonstrations_are_deterministic_and_training_only():
    examples = [derive_task_d_example(make_doc("train", f"train-{i}", i)) for i in range(8)]
    for strategy in ["uniform", "stratified_diverse"]:
        first = select_few_shot_examples_task_d(examples, 5, strategy, seed=42)
        assert len(first) == 5
        assert first == select_few_shot_examples_task_d(examples, 5, strategy, seed=42)
        assert all(ex.gold_targets[0].target == "Zug" for ex in first)
    with pytest.raises(ValueError, match="training split"):
        select_few_shot_examples_task_d([derive_task_d_example(make_doc())], 1)


def test_task_d_cli_passes_demonstrations_to_generator(tmp_path, monkeypatch):
    from germeval_llm import cli

    examples = [derive_task_d_example(make_doc("train", f"train-{i}", i)) for i in range(5)]
    monkeypatch.chdir(tmp_path)
    train_path = Path("data/derived/task_d/train.jsonl")
    train_path.parent.mkdir(parents=True)
    train_path.write_text("\n".join(ex.model_dump_json() for ex in examples))
    monkeypatch.setattr(cli, "load_canonical_jsonl", lambda _: [make_doc()])
    monkeypatch.setattr(cli, "load_gemma_model_and_tokenizer", lambda **_: (None, None))
    monkeypatch.setattr(cli, "detect_compute_device", lambda: {"device": "cpu"})
    prompts = []

    def generate(prompt):
        prompts.append(prompt)
        return '{"targets": []}', 1, 1, 1.0

    monkeypatch.setattr(cli, "build_hf_generator_fn", lambda *a, **k: generate)

    def batch_factory(*args, **kwargs):
        def batch(prompts):
            return [generate(prompt) for prompt in prompts]
        batch.validate_prompts = lambda prompts: [len(prompt) for prompt in prompts]
        batch.batch_statistics = {}
        return batch

    monkeypatch.setattr(cli, 'build_hf_batch_generator_fn', batch_factory)
    import germeval_llm.inference_checkpoint as checkpoint
    monkeypatch.setattr(checkpoint, 'model_inference_settings', lambda *a: {'model_revision': 'test'})

    # Prompt configuration paths are relative to the repository root.
    from germeval_llm import inference
    from germeval_llm.prompts import load_prompts_config
    config = load_prompts_config(Path(__file__).resolve().parents[1] / "configs/prompts.yaml")
    monkeypatch.setattr(inference, "load_prompts_config", lambda: config)
    arguments = argparse.Namespace(
        split="dev", max_docs=None, task="D", n_shots=5,
        shot_strategy="stratified_diverse", seed=42, model_id="mock",
        adapter_path=None, load_in_4bit=False, max_new_tokens=32,
        run_id="d1", prompt_key=None, model_revision="test", batch_size=2,
        checkpoint_path=None, no_checkpoint=False, no_length_grouping=False,
    )
    cli._cmd_prompt(arguments)
    cli._cmd_prompt(arguments)  # A second invocation reuses the verified checkpoint.
    assert len(prompts) == 1
    assert prompts[0].count('"target": "Zug"') == 5
    assert Path("results/predictions/d1_dev.jsonl").exists()
    manifest = json.loads(Path("results/runs/d1_dev.json").read_text())
    assert manifest["execution_mode"] == "model_inference"
    assert manifest["prompt_id"] == "d1_guidelines"
    assert manifest["n_shots"] == 5
    assert manifest["demonstration_doc_ids"] == [ex.doc_id for ex in select_few_shot_examples_task_d(examples, 5)]
    assert manifest["evaluation_doc_ids"] == ["doc"]
    assert manifest["inference_batch_statistics"]["resumed_documents"] == 1


def test_simulations_are_isolated_and_reproducible(tmp_path, monkeypatch):
    from germeval_llm import pipeline
    from germeval_llm.prompts import load_prompts_config

    config = load_prompts_config()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(pipeline, "load_prompts_config", lambda: config)
    # Keep the test independent of Java; Python scoring and file writes remain real.
    monkeypatch.setattr(pipeline, "run_official_java_evaluator", lambda *a, **k: {
        "task_c1_category_micro_f1": 0.0, "task_c2_category_sentiment_micro_f1": 0.0,
        "ote_exact_micro_f1": 0.0, "ote_overlap_micro_f1": 0.0,
    })
    plot_paths = []
    monkeypatch.setattr(pipeline, "plot_learning_curve", lambda *a, **k: plot_paths.append(k["out_path"]))
    monkeypatch.setattr(pipeline, "plot_per_category_delta_f1", lambda *a, **k: plot_paths.append(k["out_path"]))
    seeds = []
    simulate = pipeline._simulate_realistic_task_c_run

    def record_seed(*args, **kwargs):
        seeds.append((kwargs["run_id"], kwargs["split"], kwargs["seed"]))
        return simulate(*args, **kwargs)

    monkeypatch.setattr(pipeline, "_simulate_realistic_task_c_run", record_seed)
    splits = {sp: [make_doc(sp)] for sp in ["dev", "test_syn", "test_dia"]}
    sentinel = Path("results/metrics/benchmark.csv")
    sentinel.parent.mkdir(parents=True)
    sentinel.write_text("actual results\n")
    pipeline.run_m3_to_m6_benchmark_artifacts(splits)
    root = Path("results/simulations")
    first = {p.relative_to(root): p.read_bytes() for p in root.rglob("*") if p.is_file()}
    pipeline.run_m3_to_m6_benchmark_artifacts(splits)
    assert first == {p.relative_to(root): p.read_bytes() for p in root.rglob("*") if p.is_file()}
    assert sentinel.read_text() == "actual results\n"
    assert seeds[:3] == [
        ("P0_minimal_0shot", "dev", 101),
        ("P0_minimal_0shot", "test_syn", 102),
        ("P0_minimal_0shot", "test_dia", 103),
    ]
    assert all(path.startswith("results/simulations/") for path in plot_paths)
    manifest = json.loads((root / "runs/Q1_qlora_r16.json").read_text())
    assert manifest["execution_mode"] == "simulation"
    assert json.loads((root / "provenance.json").read_text())["split_seed_offsets"] == {
        "dev": 0, "test_syn": 1, "test_dia": 2,
    }


def test_notebook_specs_use_isolated_simulations():
    from germeval_llm.build_notebooks import get_notebooks_spec

    specs = get_notebooks_spec()
    for name, cells in specs.items():
        if 4 <= int(name[:2]) <= 10:
            source = "\n".join(cell.source for cell in cells)
            assert "Simulation provenance" in source
            assert "results/simulations/" in source
            assert "results/metrics/benchmark.csv" not in source
            assert "results/metrics/L1_lora_r16.json" not in source
            assert "results/metrics/learning_curves_q2.csv" not in source
    # Actual smoke runs must retain their own provenance and paths.
    assert "results/metrics/smoke_p1_dev_dev.json" in "\n".join(
        cell.source for cell in specs["04_prompting_baselines.ipynb"]
    )


def test_notebook_generation_preserves_classroom_replay(tmp_path):
    from germeval_llm.build_notebooks import build_and_execute_all_notebooks

    replay = tmp_path / "11_colab_l4_live_demo.ipynb"
    replay.write_bytes(b"recorded classroom replay and outputs\n")
    build_and_execute_all_notebooks(tmp_path)
    assert replay.read_bytes() == b"recorded classroom replay and outputs\n"
    assert len(list(tmp_path.glob("*.ipynb"))) == 12
