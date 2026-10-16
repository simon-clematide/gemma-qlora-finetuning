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
