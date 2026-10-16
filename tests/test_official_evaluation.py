"""Integration tests verifying agreement between Python evaluation.py and official EvaluationScript.jar."""

from __future__ import annotations

from pathlib import Path
import lxml.etree as ET

from germeval_llm.alignment import align_targets_to_text
from germeval_llm.evaluation import evaluate_task_c_predictions, evaluate_task_d_predictions
from germeval_llm.official_evaluation import (
    export_canonical_gold_to_official_tsv,
    export_explicit_only_gold_xml,
    export_task_c_to_official_tsv,
    export_task_d_to_official_xml,
    run_official_java_evaluator,
)
from germeval_llm.schemas import PredictionRecord, TaskDTargetItem
from germeval_llm.xml_parser import parse_document_element


def _build_fixture_docs():
    xml_str = """<Documents>
      <Document id="doc-0">
        <Opinions/>
        <relevance>false</relevance>
        <sentiment>neutral</sentiment>
        <text>Irrelevanter Text über Fußball in München.</text>
      </Document>
      <Document id="doc-1">
        <Opinions>
          <Opinion category="Zugfahrt#Haupt" from="4" to="7" target="Zug" polarity="negative"/>
          <Opinion category="Service_und_Kundenbetreuung#Haupt" from="12" to="19" target="Service" polarity="positive"/>
        </Opinions>
        <relevance>true</relevance>
        <sentiment>neutral</sentiment>
        <text>Der Zug hat Service an Bord.</text>
      </Document>
      <Document id="doc-2">
        <Opinions>
          <Opinion category="Allgemein#Haupt" from="0" to="0" target="NULL" polarity="neutral"/>
        </Opinions>
        <relevance>true</relevance>
        <sentiment>neutral</sentiment>
        <text>Ich fahre heute mit der Bahn nach Hamburg.</text>
      </Document>
    </Documents>"""
    root = ET.fromstring(xml_str)
    return [
        parse_document_element(el, source_split="test", source_ordinal=i)
        for i, el in enumerate(root.findall("Document"))
    ]


def test_official_java_evaluator_task_c_reconciliation(tmp_path: Path) -> None:
    docs = _build_fixture_docs()
    gold_tsv = export_canonical_gold_to_official_tsv(docs, tmp_path / "gold.tsv")

    # 1. Perfect predictions -> 1.0 F1 in both Python and Java
    perfect_preds = [
        PredictionRecord(
            run_id="perfect",
            task="C",
            split="test",
            doc_id=d.doc_id,
            source_ordinal=d.source_ordinal,
            raw_output="{}",
            parse_status="VALID_EXACT",
            parsed_labels=[f"{op.category}:{op.polarity}" for op in d.opinions],
        )
        for d in docs
    ]
    pred_perfect_tsv = export_task_c_to_official_tsv(docs, perfect_preds, tmp_path / "pred_perfect.tsv")
    java_perfect = run_official_java_evaluator("category", pred_perfect_tsv, gold_tsv)
    assert abs(java_perfect["task_c1_category_micro_f1"] - 1.0) < 1e-6
    assert abs(java_perfect["task_c2_category_sentiment_micro_f1"] - 1.0) < 1e-6

    # 2. Partial predictions with a false positive on empty doc-0
    partial_preds = [
        PredictionRecord(
            run_id="partial",
            task="C",
            split="test",
            doc_id="doc-0",
            source_ordinal=0,
            raw_output="{}",
            parse_status="VALID_EXACT",
            parsed_labels=["Allgemein:negative"],  # FP on empty gold doc!
        ),
        PredictionRecord(
            run_id="partial",
            task="C",
            split="test",
            doc_id="doc-1",
            source_ordinal=1,
            raw_output="{}",
            parse_status="VALID_EXACT",
            parsed_labels=["Zugfahrt:negative", "Ticketkauf:negative"],  # 1 TP, 1 FP, 1 FN
        ),
        PredictionRecord(
            run_id="partial",
            task="C",
            split="test",
            doc_id="doc-2",
            source_ordinal=2,
            raw_output="{}",
            parse_status="VALID_EXACT",
            parsed_labels=[],  # 1 FN
        ),
    ]
    pred_partial_tsv = export_task_c_to_official_tsv(docs, partial_preds, tmp_path / "pred_partial.tsv")
    java_partial = run_official_java_evaluator("category", pred_partial_tsv, gold_tsv)
    py_partial = evaluate_task_c_predictions(docs, partial_preds, n_bootstrap=50)

    # Verify exact numerical agreement between Python java_compat mode and EvaluationScript.jar
    assert abs(
        py_partial["task_c1_category_java_compat"]["micro_f1"] - java_partial["task_c1_category_micro_f1"]
    ) < 1e-6
    assert abs(
        py_partial["task_c2_category_sentiment_java_compat"]["micro_f1"]
        - java_partial["task_c2_category_sentiment_micro_f1"]
    ) < 1e-6
    # Standard F1 = 2*1 / (2*1 + 2 + 2) = 1/3; Java Fscore.java (including 'none' pseudo-label) = 1 / (1 + 2 + 2) = 0.2
    assert abs(py_partial["task_c2_category_sentiment_strict"]["micro_f1"] - (1.0 / 3.0)) < 1e-6
    assert abs(py_partial["task_c2_category_sentiment_java_compat"]["micro_f1"] - 0.2) < 1e-6


def test_official_java_evaluator_task_d_ote_reconciliation(tmp_path: Path) -> None:
    docs = _build_fixture_docs()
    gold_xml = export_explicit_only_gold_xml(docs, tmp_path / "gold_explicit.xml")

    d_preds = []
    for d in docs:
        if d.doc_id == "doc-1":
            items = [TaskDTargetItem(target="Zug", category="Zugfahrt", polarity="negative")]
        else:
            items = []
        aligned = align_targets_to_text(d.text, items)
        d_preds.append(
            PredictionRecord(
                run_id="d_partial",
                task="D",
                split="test",
                doc_id=d.doc_id,
                source_ordinal=d.source_ordinal,
                raw_output="{}",
                parse_status="VALID_EXACT",
                parsed_targets=items,
                aligned_spans=aligned,
            )
        )

    pred_xml = export_task_d_to_official_xml(docs, d_preds, tmp_path / "pred_d.xml")
    java_ote = run_official_java_evaluator("OTE", pred_xml, gold_xml)
    py_ote = evaluate_task_d_predictions(docs, d_preds)

    # 1 TP ("Zug"), 0 FP, 1 FN ("Service"):
    # Standard F1 = 2*1 / (2*1 + 0 + 1) = 2/3 = 0.666667
    # Official Java Fscore.java (includes "missmatch" as a category in micro sum) = 1 / (1 + 0 + 1) = 0.5
    assert abs(py_ote["span_explicit_exact"]["f1"] - (2.0 / 3.0)) < 1e-6
    assert abs(java_ote["ote_exact_micro_f1"] - py_ote["span_explicit_exact_java_compat"]["micro_f1"]) < 1e-6
    assert abs(java_ote["ote_overlap_micro_f1"] - py_ote["span_explicit_overlap_java_compat"]["micro_f1"]) < 1e-6
