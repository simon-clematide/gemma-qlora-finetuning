"""Unit tests for Task C parsing, Gemma chat template formatting, completion-only loss masking, and metrics."""

from __future__ import annotations

from germeval_llm.evaluation import evaluate_label_sets
from germeval_llm.parsing import parse_task_c_output
from germeval_llm.prompts import (
    compute_completion_only_labels,
    format_gemma_chat_string,
)
from germeval_llm.schemas import ChatMessage


def test_parse_task_c_output_variants() -> None:
    # Exact valid JSON
    labels, status, errors = parse_task_c_output('{"labels": ["Zugfahrt:negative", "Allgemein:neutral", "Zugfahrt:negative"]}')
    assert status == "VALID_EXACT"
    assert errors == []
    assert labels == ["Allgemein:neutral", "Zugfahrt:negative"]

    # Markdown fenced JSON
    labels_f, status_f, errors_f = parse_task_c_output('```json\n{"labels": []}\n```')
    assert status_f == "VALID_FENCED"
    assert errors_f == []
    assert labels_f == []

    # Unknown category rejected (not silently mapped)
    labels_u, status_u, errors_u = parse_task_c_output('{"labels": ["UnknownCategory:negative", "Ticketkauf:positive"]}')
    assert status_u == "UNKNOWN_LABEL"
    assert len(errors_u) == 1
    assert labels_u == ["Ticketkauf:positive"]

    # Malformed JSON
    labels_bad, status_bad, _ = parse_task_c_output('{"labels": ["Zugfahrt:negative"')
    assert status_bad == "INVALID_JSON"
    assert labels_bad == []


def test_gemma_chat_template_and_completion_only_loss_masking() -> None:
    prompt_msgs = [
        ChatMessage(role="system", content="System-Instruktion."),
        ChatMessage(role="user", content="Text: Der Zug hat Verspätung."),
    ]
    comp_msgs = [ChatMessage(role="assistant", content='{"labels":["Zugfahrt:negative"]}')]

    rendered = format_gemma_chat_string(prompt_msgs, comp_msgs, include_bos=True)
    # Single <bos>, system merged into user turn, model turn present
    assert rendered.count("<bos>") == 1
    assert "<start_of_turn>system" not in rendered
    assert "<start_of_turn>user\nSystem-Instruktion.\n\nText: Der Zug hat Verspätung.<end_of_turn>\n" in rendered
    assert rendered.endswith('<start_of_turn>model\n{"labels":["Zugfahrt:negative"]}<end_of_turn>\n')

    # Verify completion-only loss masking at byte/character token level
    marker = "<start_of_turn>model\n"
    input_ids = [ord(c) for c in rendered]
    marker_ids = [ord(c) for c in marker]
    labels = compute_completion_only_labels(input_ids, marker_ids, ignore_index=-100)

    marker_end_pos = rendered.index(marker) + len(marker)
    assert all(x == -100 for x in labels[:marker_end_pos])
    assert labels[marker_end_pos:] == [ord(c) for c in '{"labels":["Zugfahrt:negative"]}<end_of_turn>\n']


def test_strict_vs_java_compat_empty_gold_document_scoring() -> None:
    # Document 0 has empty gold (e.g. relevance=false), Document 1 has 1 gold label
    gold_sets = [set(), {"Zugfahrt:negative"}]
    # Model predicts a false positive on Doc 0, and a true positive on Doc 1
    pred_sets = [{"Allgemein:neutral"}, {"Zugfahrt:negative"}]

    strict_res = evaluate_label_sets(gold_sets, pred_sets, ignore_empty_gold_docs=False)
    assert strict_res["tp"] == 1
    assert strict_res["fp"] == 1
    assert strict_res["fn"] == 0
    assert abs(strict_res["micro_f1"] - (2.0 / 3.0)) < 1e-6

    java_res = evaluate_label_sets(gold_sets, pred_sets, ignore_empty_gold_docs=True)
    # Java evaluator skips Doc 0 because gold is empty!
    assert java_res["tp"] == 1
    assert java_res["fp"] == 0
    assert java_res["fn"] == 0
    assert abs(java_res["micro_f1"] - 1.0) < 1e-6


def test_align_predictions_rejects_duplicate_missing_and_unexpected_ids() -> None:
    import pytest
    from germeval_llm.evaluation import align_predictions
    from germeval_llm.schemas import CanonicalDocument, PredictionRecord

    docs = [
        CanonicalDocument(
            doc_id="doc_1",
            source_split="test_syn",
            source_ordinal=0,
            text="Text 1",
            relevance=True,
            document_sentiment="neutral",
            opinions=[],
        ),
        CanonicalDocument(
            doc_id="doc_2",
            source_split="test_syn",
            source_ordinal=1,
            text="Text 2",
            relevance=True,
            document_sentiment="neutral",
            opinions=[],
        ),
    ]

    def _make_pred(doc_id: str, split: str = "test_syn") -> PredictionRecord:
        return PredictionRecord(
            run_id="r1",
            task="C",
            split=split,
            doc_id=doc_id,
            source_ordinal=0,
            raw_output='{"labels":[]}',
            parse_status="VALID_EXACT",
            parsed_labels=[],
        )

    # Valid alignment (even when out of order in predictions list)
    aligned = align_predictions(docs, [_make_pred("doc_2"), _make_pred("doc_1")], expected_split="test_syn")
    assert [p.doc_id for p in aligned] == ["doc_1", "doc_2"]

    # 1. Duplicate prediction IDs -> explicit ValueError
    with pytest.raises(ValueError, match="Duplicate prediction document IDs"):
        align_predictions(docs, [_make_pred("doc_1"), _make_pred("doc_1")])

    # 2. Missing prediction IDs -> explicit ValueError
    with pytest.raises(ValueError, match="Prediction mismatch: 1 missing, 0 unexpected"):
        align_predictions(docs, [_make_pred("doc_1")])

    # 3. Unexpected prediction IDs -> explicit ValueError
    with pytest.raises(ValueError, match="Prediction mismatch: 0 missing, 1 unexpected"):
        align_predictions(docs, [_make_pred("doc_1"), _make_pred("doc_2"), _make_pred("doc_extra")])

    # 4. Split mismatch -> explicit ValueError
    with pytest.raises(ValueError, match="Prediction split mismatch"):
        align_predictions(docs, [_make_pred("doc_1", split="dev"), _make_pred("doc_2", split="dev")], expected_split="test_syn")


def test_verify_experiment_manifest_blocks_mismatches() -> None:
    import pytest
    from germeval_llm.evaluation import verify_experiment_manifest
    from germeval_llm.schemas import CanonicalDocument

    docs = [
        CanonicalDocument(
            doc_id="d1",
            source_split="test_syn",
            source_ordinal=0,
            text="Bahn",
            relevance=False,
            document_sentiment="neutral",
            opinions=[],
        )
    ]
    valid_manifest = {
        "experiment_id": "classroom_fast_n1000",
        "base_model": "google/gemma-2-2b-it",
        "prompt_id": "p1_guidelines",
        "training_split": "train",
        "evaluation_split": "test_syn",
        "evaluation_num_docs": 1,
        "evaluation_doc_ids": ["d1"],
        "seed": 42,
    }
    assert verify_experiment_manifest(valid_manifest, "test_syn", "p1_guidelines", expected_eval_docs=docs) == valid_manifest

    # Mismatched evaluation split
    with pytest.raises(ValueError, match="Replay manifest mismatch: evaluation_split"):
        verify_experiment_manifest(valid_manifest, "dev", "p1_guidelines", expected_eval_docs=docs)

    # Mismatched prompt_id
    with pytest.raises(ValueError, match="Replay manifest mismatch: prompt_id"):
        verify_experiment_manifest(valid_manifest, "test_syn", "p0_minimal", expected_eval_docs=docs)

    # Mismatched evaluation document IDs
    bad_manifest = dict(valid_manifest, evaluation_doc_ids=["other_doc"])
    with pytest.raises(ValueError, match="Replay manifest mismatch: evaluation_doc_ids"):
        verify_experiment_manifest(bad_manifest, "test_syn", "p1_guidelines", expected_eval_docs=docs)


def test_missing_qualitative_archetype_graceful_fallback() -> None:
    from germeval_llm.reporting import select_pedagogical_error_analysis_examples
    from germeval_llm.schemas import CanonicalDocument, CanonicalOpinion, PredictionRecord

    # Only 1 document where both models are exact match -> other 4 archetypes are unavailable
    doc = CanonicalDocument(
        doc_id="doc_only",
        source_split="test_syn",
        source_ordinal=0,
        text="Der Zug war pünktlich.",
        relevance=True,
        document_sentiment="positive",
        opinions=[
            CanonicalOpinion(
                category_full="Zugfahrt#Haupt",
                category="Zugfahrt",
                subcategory="Haupt",
                polarity="positive",
                target_xml="Zug",
                start_xml=4,
                end_xml=7,
                is_implicit=False,
                is_discontinuous_star=False,
            )
        ],
    )
    b_pred = PredictionRecord(
        run_id="b",
        task="C",
        split="test_syn",
        doc_id="doc_only",
        source_ordinal=0,
        raw_output='{"labels":["Zugfahrt:positive"]}',
        parse_status="VALID_EXACT",
        parsed_labels=["Zugfahrt:positive"],
    )
    q_pred = PredictionRecord(
        run_id="q",
        task="C",
        split="test_syn",
        doc_id="doc_only",
        source_ordinal=0,
        raw_output='{"labels":["Zugfahrt:positive"]}',
        parse_status="VALID_EXACT",
        parsed_labels=["Zugfahrt:positive"],
    )

    rows = select_pedagogical_error_analysis_examples([doc], [b_pred], [q_pred])
    assert len(rows) == 5
    avail_map = {r["archetype"]: r["available"] for r in rows}
    assert avail_map["2. Both models correct"] is True
    assert avail_map["1. Baseline error corrected"] is False
    assert avail_map["4. Fine-tuning introduces error"] is False
    assert avail_map["5. Output-format / schema problem"] is False
    for r in rows:
        if not r["available"]:
            assert r["doc_id"] == "UNAVAILABLE"


def test_completion_only_masking_end_of_turn_and_truncation() -> None:
    import pytest

    marker = "<start_of_turn>model\n"
    completion = '{"labels":["Zugfahrt:positive"]}<end_of_turn>\n'
    full_str = f"<bos><start_of_turn>user\nInstruktion und Text<end_of_turn>\n{marker}{completion}"

    input_ids = [ord(c) for c in full_str]
    marker_ids = [ord(c) for c in marker]

    # 1. Partial truncation inside the completion preserves remaining completion tokens
    cutoff = len(input_ids) - 5
    labels_partial = compute_completion_only_labels(input_ids, marker_ids, max_seq_length=cutoff)
    assert len(labels_partial) == cutoff
    prompt_len = full_str.index(marker) + len(marker)
    assert all(x == -100 for x in labels_partial[:prompt_len])
    assert labels_partial[prompt_len:] == input_ids[prompt_len:cutoff]

    # 2. Truncation right at the end of the response marker leaves 0 completion tokens -> raises ValueError
    with pytest.raises(ValueError, match="No completion tokens remain"):
        compute_completion_only_labels(input_ids, marker_ids, max_seq_length=prompt_len)

    # 3. Truncation before the response marker completes -> raises ValueError
    with pytest.raises(ValueError, match="Response template marker not found"):
        compute_completion_only_labels(input_ids, marker_ids, max_seq_length=prompt_len - 3)


def test_extract_training_and_eval_series_and_paired_bootstrap() -> None:
    from germeval_llm.evaluation import paired_bootstrap_delta_f1_ci
    from germeval_llm.training import extract_training_and_eval_series

    raw_log = [
        {"step": 10, "loss": 1.45, "learning_rate": 2e-4},
        {"step": 20, "loss": 0.82, "learning_rate": 1.8e-4},
        {"step": 33, "eval_loss": 0.54},
        {"step": 40, "loss": 0.41, "learning_rate": 1.2e-4},
    ]
    train_s, eval_s = extract_training_and_eval_series(raw_log)
    assert [e["step"] for e in train_s] == [10, 20, 40]
    assert [e["step"] for e in eval_s] == [33]

    gold = [{"Zugfahrt:positive"}, {"Service_und_Kundenbetreuung:negative"}, set(), {"Allgemein:neutral"}]
    base = [set(), {"Service_und_Kundenbetreuung:positive"}, {"Allgemein:neutral"}, {"Allgemein:neutral"}]
    ft = [{"Zugfahrt:positive"}, {"Service_und_Kundenbetreuung:negative"}, set(), {"Allgemein:neutral"}]

    ci = paired_bootstrap_delta_f1_ci(gold, base, ft, n_bootstrap=200, seed=42)
    assert ci["point_ft_f1"] > ci["point_baseline_f1"]
    assert ci["point_delta_f1"] > 0.0
    assert ci["delta_ci95_low"] <= ci["point_delta_f1"] <= ci["delta_ci95_high"]


def test_colab_live_demo_notebook_replay_integrity() -> None:
    import json
    from pathlib import Path
    import nbformat
    from germeval_llm.evaluation import compare_task_c_models, verify_experiment_manifest
    from germeval_llm.inference import load_predictions_jsonl
    from germeval_llm.xml_parser import load_canonical_jsonl

    nb_path = Path("notebooks/11_colab_l4_live_demo.ipynb")
    assert nb_path.exists()
    nb = nbformat.read(str(nb_path), as_version=4)

    # Verify all code cells executed cleanly without error outputs and rendered both PNG figures
    png_count = 0
    for cell in nb.cells:
        if cell.cell_type == "code":
            assert cell.outputs, "Expected pre-executed outputs in 11_colab_l4_live_demo.ipynb"
            for out in cell.outputs:
                assert out.output_type != "error", f"Cell raised error: {out}"
                if "image/png" in out.get("data", {}):
                    png_count += 1
    assert png_count >= 2, "Expected loss curve and before/after comparison PNG figures"

    # Verify the portable fresh recording and reproduce its stored comparison.
    from germeval_llm.replay import DEFAULT_CLASSROOM_REPLAY, load_replay_manifest
    manifest = load_replay_manifest()
    assert manifest["hardware"]["device"] == "mps"
    assert manifest["method"] == "LoRA" and manifest["quantization"] == "none"
    assert manifest["weights_required_for_replay"] is False
    eval_docs = load_canonical_jsonl("data/canonical/test_syn.jsonl")[: manifest["evaluation_num_docs"]]
    verify_experiment_manifest(
        manifest,
        expected_split="test_syn",
        expected_prompt_id="p1_guidelines",
        expected_eval_docs=eval_docs,
        expected_base_model="google/gemma-2-2b-it",
    )
    b_preds = load_predictions_jsonl(manifest["baseline_prediction_file"])
    q_preds = load_predictions_jsonl(manifest["finetuned_prediction_file"])
    comp = compare_task_c_models(
        eval_docs, b_preds, q_preds, expected_split="test_syn",
        apply_gold_typo_patches=manifest["gold_typo_patches"], seed=manifest["seed"],
    )
    expected = json.loads((DEFAULT_CLASSROOM_REPLAY.parent / "comparison.json").read_text())
    assert comp == expected
    rendered = "\n".join(
        "".join(out.get("text", [])) + "".join(out.get("data", {}).get("text/plain", []))
        for cell in nb.cells if cell.cell_type == "code" for out in cell.outputs
    )
    assert manifest["experiment_id"] in rendered
    assert "Recorded MPS LoRA (bfloat16)" in rendered
    assert "0.5985" in rendered and "0.1008" in rendered
    assert "QLoRA (4-bit NF4)" not in rendered
    assert "classroom_fast_n1000_seed42" not in rendered


def test_consistent_fences_do_not_reduce_label_scores() -> None:
    import lxml.etree as ET
    from germeval_llm.evaluation import evaluate_task_c_predictions
    from germeval_llm.schemas import PredictionRecord
    from germeval_llm.xml_parser import parse_document_element

    doc = parse_document_element(ET.fromstring(
        '<Document id="d"><text>Der Zug fällt aus.</text><relevance>true</relevance>'
        '<sentiment>negative</sentiment><Opinions><Opinion category="Zugfahrt#Haupt" '
        'polarity="negative" target="Zug" from="4" to="7"/></Opinions></Document>'
    ), source_split="dev", source_ordinal=0)
    body = '{"labels": ["Zugfahrt:negative"]}'
    metrics = []
    for raw in [body, f'```json\n{body}\n```']:
        labels, status, errors = parse_task_c_output(raw)
        prediction = PredictionRecord(
            run_id="wrapper-test", task="C", split="dev", doc_id="d", source_ordinal=0,
            raw_output=raw, parsed_labels=labels, parse_status=status, parse_errors=errors,
        )
        metrics.append(evaluate_task_c_predictions([doc], [prediction], n_bootstrap=10))
    assert metrics[0]["task_c2_category_sentiment_strict"] == metrics[1]["task_c2_category_sentiment_strict"]
    assert metrics[0]["task_c2_category_sentiment_strict"]["micro_f1"] == 1.0
    assert metrics[0]["schema_validity_rate"] == metrics[1]["schema_validity_rate"] == 1.0
    assert metrics[0]["valid_exact_rate"] == 1.0 and metrics[1]["valid_exact_rate"] == 0.0


def test_demo_access_blocks_before_setup_and_skips_replay(monkeypatch) -> None:
    """Exercise the first cell without credentials, network, or an ML installation."""
    import importlib.util
    import sys
    from types import ModuleType, SimpleNamespace

    import nbformat
    import pytest

    nb = nbformat.read('notebooks/11_colab_l4_live_demo.ipynb', as_version=4)
    access = next(cell.source for cell in nb.cells if cell.cell_type == 'code')
    calls = []
    hub = ModuleType('huggingface_hub')
    saved_token = [None]
    hub.get_token = lambda: saved_token[0]

    def login(**kwargs):
        calls.append('login')
        saved_token[0] = kwargs['token']

    hub.login = login
    hub.HfApi = lambda: SimpleNamespace(auth_check=lambda *args, **kwargs: calls.append('check'))
    torch = ModuleType('torch')
    torch.cuda = SimpleNamespace(is_available=lambda: True)
    monkeypatch.setitem(sys.modules, 'huggingface_hub', hub)
    monkeypatch.setitem(sys.modules, 'torch', torch)
    original_find_spec = importlib.util.find_spec
    monkeypatch.setattr(importlib.util, 'find_spec', lambda name: (
        True if name in {'torch', 'huggingface_hub'} else original_find_spec(name)
    ))
    monkeypatch.setattr('getpass.getpass', lambda prompt: calls.append('hidden prompt') or 'test-only-token')

    scope = {}
    exec(access, scope)
    assert scope['RUN_LIVE_GPU'] is True
    assert calls == ['hidden prompt', 'check', 'login']
    assert '_access_token' not in scope

    calls.clear()
    saved_token[0] = 'test-only-saved-token'
    exec(access, {})
    assert calls == ['check', 'login']

    calls.clear()
    def check_replacement(*args, **kwargs):
        calls.append('check')
        if kwargs['token'] == 'test-only-saved-token':
            raise ValueError('sensitive error detail')

    hub.HfApi = lambda: SimpleNamespace(auth_check=check_replacement)
    exec(access, {})
    assert calls == ['check', 'hidden prompt', 'check', 'login']

    # A Colab secret may still override login's saved replacement.
    saved_token[0] = 'test-only-saved-token'
    hub.login = lambda **kwargs: calls.append('login')
    scope = {}
    with pytest.raises(RuntimeError, match='overrides the token'):
        exec(access, scope)
    assert '_access_token' not in scope

    def denied(*args, **kwargs):
        raise ValueError('sensitive error detail')

    hub.HfApi = lambda: SimpleNamespace(auth_check=denied)
    scope = {}
    with pytest.raises(RuntimeError, match='ValueError') as failure:
        exec(access, scope)
    assert 'sensitive error detail' not in str(failure.value)
    assert 'Setup has not started' in str(failure.value)
    assert '_access_token' not in scope

    calls.clear()
    scope = {}
    exec(access.replace("REQUESTED_MODE = 'auto'", "REQUESTED_MODE = 'replay'", 1), scope)
    assert scope['RUN_LIVE_GPU'] is False
    assert calls == []
