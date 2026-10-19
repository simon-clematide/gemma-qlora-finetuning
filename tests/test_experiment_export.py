import json
from zipfile import ZipFile

import nbformat
import pytest

from germeval_llm.experiment_export import export_live_experiment, verify_export_bundle
from germeval_llm.schemas import CanonicalDocument, PredictionRecord


def test_live_export_round_trip_and_provenance(tmp_path):
    import matplotlib
    matplotlib.use("Agg")

    # Small synthetic fixture, never committed as a measured experiment.
    doc = CanonicalDocument(doc_id="export-test", source_split="test_syn", source_ordinal=0,
                            text="No opinion", relevance=False, document_sentiment="neutral", opinions=[])
    pred = PredictionRecord(run_id="test", task="C", split="test_syn", doc_id=doc.doc_id,
                            source_ordinal=0, raw_output='{"labels": []}', parsed_labels=[],
                            parse_status="VALID_EXACT")
    summary = {
        "device": "cuda", "hardware": {"device": "cuda", "gpu_name": "test GPU"},
        "sequence_length_audit": {"chosen_limit": 512, "measured_max": 500},
        "loaded_in_4bit": True, "base_model_id": "test-model", "model_revision": "test-revision",
        "trainable_parameter_dtypes": ["torch.float32"], "seed": 7,
        "tokenized_training_doc_ids": ["train-only"], "tokenized_validation_doc_ids": ["dev-only"],
        "actual_training_settings": {"global_step": 2, "max_length": 512, "effective_batch_size": 4},
        "log_history": [{"step": 1, "loss": 1.2}, {"step": 2, "loss": 0.8, "eval_loss": 0.9}],
    }
    args = dict(output_root=tmp_path / "exports", eval_docs=[doc], baseline_predictions=[pred],
                finetuned_predictions=[pred], training_summary=summary, training_config="configs/qlora.yaml")
    archive = export_live_experiment(**args)
    with ZipFile(archive) as z:
        assert all(name.startswith(archive.stem + "/") for name in z.namelist())
        z.extractall(tmp_path / "imported")
    bundle = tmp_path / "imported" / archive.stem
    manifest = verify_export_bundle(bundle)
    assert manifest["execution_mode"] == "live"
    assert manifest["sequence_length_audit"] == summary["sequence_length_audit"]
    assert manifest["hardware"]["gpu_name"] == "test GPU"
    assert manifest["quantization"] == "nf4_4bit"
    assert manifest["training_doc_ids"] == ["train-only"]
    assert manifest["evaluation_doc_ids"] == [doc.doc_id]
    assert manifest["max_steps"] == 2
    assert manifest["model_revision"] == "test-revision"
    assert "source_run" not in manifest and "recording_integrity_verification" not in manifest
    assert json.loads((bundle / "baseline_predictions.jsonl").read_text())["raw_output"] == pred.raw_output
    assert (bundle / "comparison.png").read_bytes().startswith(b"\x89PNG")
    assert not any(p.suffix in {".safetensors", ".bin"} for p in bundle.iterdir())
    (bundle / "comparison.json").write_text("{}")
    with pytest.raises(ValueError, match="hash mismatch"):
        verify_export_bundle(bundle)
    with pytest.raises(ValueError, match="missing"):
        export_live_experiment(**dict(args, baseline_predictions=[]))
    with pytest.raises(ValueError, match="measured CUDA"):
        export_live_experiment(**dict(args, training_summary=dict(summary, device="mps")))
    second = export_live_experiment(**dict(args, training_summary=dict(summary, loaded_in_4bit=False), sampling_strategy="all", inference_batch_size=8))
    assert second != archive
    second_manifest = verify_export_bundle(second.with_suffix(""))
    assert second_manifest["quantization"] == "none"
    assert second_manifest["sampling_strategy"] == "all"
    assert second_manifest["generation"]["batch_size"] == 8
    assert second_manifest["generation"]["latency_semantics"] == "amortized batch wall time per document"


def test_notebook_export_skips_replay_and_downloads_live(monkeypatch, tmp_path):
    import sys
    from types import ModuleType

    import germeval_llm.experiment_export as exporter

    nb = nbformat.read("notebooks/11_colab_l4_live_demo.ipynb", as_version=4)
    source = nb.cells[-1].source
    exec(source, {"RUN_LIVE_GPU": False})
    calls = []
    archive = tmp_path / "run.zip"
    monkeypatch.setattr(exporter, "export_live_experiment", lambda **kw: calls.append(kw) or archive)
    colab = ModuleType("google.colab")
    colab.files = ModuleType("google.colab.files")
    colab.files.download = lambda name: calls.append(name)
    monkeypatch.setitem(sys.modules, "google.colab", colab)
    exec(source, dict(RUN_LIVE_GPU=True, sys=sys, eval_docs=[], baseline_preds=[], ft_preds=[],
                     train_summary={}, TRAIN_CONFIG="config", INFERENCE_BATCH_SIZE=8, exp_manifest={"gold_typo_patches": False}))
    assert calls[-1] == str(archive)
