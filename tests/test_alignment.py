"""Unit tests for Task D target-string parsing, deterministic character-span alignment, and Task D evaluation."""

from __future__ import annotations

import lxml.etree as ET

from germeval_llm.alignment import align_targets_to_text
from germeval_llm.evaluation import evaluate_task_d_predictions
from germeval_llm.parsing import parse_task_d_output
from germeval_llm.schemas import PredictionRecord, TaskDTargetItem
from germeval_llm.xml_parser import parse_document_element


def test_task_d_output_parser_rejects_offsets() -> None:
    # Valid target-only output
    targets, status, errors = parse_task_d_output(
        '{"targets": [{"target": "Zug", "category": "Zugfahrt", "polarity": "negative"}]}'
    )
    assert status == "VALID_EXACT"
    assert errors == []
    assert len(targets) == 1

    # Output with forbidden offset field must be rejected
    targets_bad, status_bad, errors_bad = parse_task_d_output(
        '{"targets": [{"target": "Zug", "category": "Zugfahrt", "polarity": "negative", "from": 4, "to": 7}]}'
    )
    assert status_bad == "INVALID_SCHEMA"
    assert targets_bad == []
    assert any("forbidden offset keys" in e for e in errors_bad)


def test_deterministic_alignment_unique_ambiguous_discontinuous_and_unaligned() -> None:
    text = "Der Zug fällt aus. Der Zug ist überfüllt. Er zeigt falsche Abfahrtszeiten an."
    preds = [
        TaskDTargetItem(target="Zug", category="Zugfahrt", polarity="negative"),
        TaskDTargetItem(target="Zug", category="Zugfahrt", polarity="negative"),
        TaskDTargetItem(target="zeigt * an", category="Informationen", polarity="negative"),
        TaskDTargetItem(target="Bordbistro", category="Gastronomisches_Angebot", polarity="neutral"),
    ]
    spans = align_targets_to_text(text, preds)
    assert len(spans) == 4

    # First "Zug" -> occurrence 0 (4..7)
    assert spans[0].alignment_status == "ambiguous_contiguous"
    assert (spans[0].start_char, spans[0].end_char) == (4, 7)
    assert spans[0].assigned_occurrence_index == 0

    # Second "Zug" with same category -> advances to occurrence 1 (23..26)
    assert spans[1].alignment_status == "ambiguous_contiguous"
    assert (spans[1].start_char, spans[1].end_char) == (23, 26)
    assert spans[1].assigned_occurrence_index == 1

    # Discontinuous "zeigt * an" -> outer bounding span covering "zeigt falsche Abfahrtszeiten an" (45..76)
    assert spans[2].alignment_status == "discontinuous_star"
    assert text[spans[2].start_char : spans[2].end_char] == "zeigt falsche Abfahrtszeiten an"

    # Missing "Bordbistro" -> unaligned (None, None)
    assert spans[3].alignment_status == "unaligned"
    assert spans[3].start_char is None and spans[3].end_char is None


def test_evaluate_task_d_predictions_counts_unaligned_as_fp() -> None:
    doc = parse_document_element(
        ET.fromstring(
            """
            <Document id="doc-d1">
              <Opinions>
                <Opinion category="Zugfahrt#Haupt" from="4" to="7" target="Zug" polarity="negative"/>
              </Opinions>
              <relevance>true</relevance>
              <sentiment>negative</sentiment>
              <text>Der Zug hat Verspätung.</text>
            </Document>
            """
        ),
        source_split="dev",
        source_ordinal=0,
    )
    pred_targets = [
        TaskDTargetItem(target="Zug", category="Zugfahrt", polarity="negative"),
        TaskDTargetItem(target="Halluzination", category="Allgemein", polarity="negative"),
    ]
    aligned = align_targets_to_text(doc.text, pred_targets)
    rec = PredictionRecord(
        run_id="test_d",
        task="D",
        split="dev",
        doc_id="doc-d1",
        source_ordinal=0,
        raw_output="{}",
        parse_status="VALID_EXACT",
        parsed_targets=pred_targets,
        aligned_spans=aligned,
    )
    metrics = evaluate_task_d_predictions([doc], [rec])
    # 1 TP ("Zug") and 1 FP ("Halluzination" even though unaligned) -> Precision 0.5, Recall 1.0, F1 2/3
    assert metrics["span_explicit_exact"]["tp"] == 1
    assert metrics["span_explicit_exact"]["fp"] == 1
    assert metrics["span_explicit_exact"]["fn"] == 0
    assert abs(metrics["span_explicit_exact"]["f1"] - (2.0 / 3.0)) < 1e-6
