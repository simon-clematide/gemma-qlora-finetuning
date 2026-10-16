"""Export predictions to official GermEval 2017 TSV/XML formats and run EvaluationScript.jar."""

from __future__ import annotations

from pathlib import Path
import re
import shutil
import subprocess
from typing import Any, Sequence
import lxml.etree as ET

from germeval_llm.schemas import CanonicalDocument, PredictionRecord

JAVA8_MACOS_PATH = "/Library/Internet Plug-Ins/JavaAppletPlugin.plugin/Contents/Home/bin/java"
DEFAULT_JAR_PATH = Path("data/raw/EvaluationScript.jar")

_MICRO_F1_RE = re.compile(r"MICRO_F1\s+([0-9.]+|NaN)")
_INSTANCES_RE = re.compile(r"\(based on\s+(\d+)\s+instances\)")


def find_java_binary(require_jaxb: bool = False) -> str:
    """Locate a working Java binary.

    When evaluating XML files (`OTE`), `EvaluationScript.jar` requires `javax.xml.bind` (built-in
    in Java 8). On macOS with Java 8 installed at JAVA8_MACOS_PATH, prefer it when `require_jaxb=True`.
    """
    if require_jaxb and Path(JAVA8_MACOS_PATH).exists():
        return JAVA8_MACOS_PATH
    if Path(JAVA8_MACOS_PATH).exists():
        return JAVA8_MACOS_PATH
    java_bin = shutil.which("java")
    if not java_bin:
        raise RuntimeError("No 'java' executable found on PATH.")
    return java_bin


def export_task_c_to_official_tsv(
    canonical_docs: Sequence[CanonicalDocument],
    predictions: Sequence[PredictionRecord],
    out_tsv_path: str | Path,
) -> Path:
    """Export Task C predictions to the 5-column TSV format expected by `GermevalReader.readTSV`:
    `DOC_ID \\t TEXT \\t RELEVANCE \\t SENTIMENT \\t CAT1:POL1 CAT2:POL2`
    """
    pred_by_id = {p.doc_id: p for p in predictions}
    out = Path(out_tsv_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    with out.open("w", encoding="utf-8") as f:
        for doc in canonical_docs:
            pred = pred_by_id.get(doc.doc_id)
            labels = pred.parsed_labels if pred else []
            clean_text = doc.text.replace("\t", " ").replace("\n", " ").replace("\r", " ")
            rel_str = "true" if doc.relevance else "false"
            sent_str = doc.document_sentiment or "neutral"
            labels_col = " ".join(labels)
            if labels_col:
                f.write(f"{doc.doc_id}\t{clean_text}\t{rel_str}\t{sent_str}\t{labels_col}\t\n")
            else:
                f.write(f"{doc.doc_id}\t{clean_text}\t{rel_str}\t{sent_str}\t\n")
    return out


def export_canonical_gold_to_official_tsv(
    canonical_docs: Sequence[CanonicalDocument],
    out_tsv_path: str | Path,
) -> Path:
    """Export canonical gold annotations to official 5-column TSV format for `EvaluationScript.jar category`."""
    out = Path(out_tsv_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for doc in canonical_docs:
            clean_text = doc.text.replace("\t", " ").replace("\n", " ").replace("\r", " ")
            rel_str = "true" if doc.relevance else "false"
            sent_str = doc.document_sentiment or "neutral"
            labels = [f"{op.category}:{op.polarity}" for op in doc.opinions]
            labels_col = " ".join(labels)
            if labels_col:
                f.write(f"{doc.doc_id}\t{clean_text}\t{rel_str}\t{sent_str}\t{labels_col}\t\n")
            else:
                f.write(f"{doc.doc_id}\t{clean_text}\t{rel_str}\t{sent_str}\t\n")
    return out


def export_task_d_to_official_xml(
    canonical_docs: Sequence[CanonicalDocument],
    predictions: Sequence[PredictionRecord],
    out_xml_path: str | Path,
) -> Path:
    """Export aligned Task D predictions to official GermEval 2017 XML format for `EvaluationScript.jar OTE`.

    Only successfully aligned predictions (`start_char is not None and end_char is not None`) are written
    as `<Opinion>` elements.
    """
    pred_by_id = {p.doc_id: p for p in predictions}
    root = ET.Element("Documents")

    for doc in canonical_docs:
        doc_elem = ET.SubElement(root, "Document", attrib={"id": doc.doc_id})
        pred = pred_by_id.get(doc.doc_id)
        aligned_spans = [
            sp
            for sp in (pred.aligned_spans if pred else [])
            if sp.start_char is not None and sp.end_char is not None
        ]
        if aligned_spans:
            ops_elem = ET.SubElement(doc_elem, "Opinions")
            for sp in aligned_spans:
                ET.SubElement(
                    ops_elem,
                    "Opinion",
                    attrib={
                        "category": sp.category,
                        "from": str(sp.start_char),
                        "to": str(sp.end_char),
                        "target": sp.target,
                        "polarity": sp.polarity,
                    },
                )
        rel_elem = ET.SubElement(doc_elem, "relevance")
        rel_elem.text = "true" if doc.relevance else "false"
        sent_elem = ET.SubElement(doc_elem, "sentiment")
        sent_elem.text = doc.document_sentiment or "neutral"
        text_elem = ET.SubElement(doc_elem, "text")
        text_elem.text = doc.text

    out = Path(out_xml_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    tree = ET.ElementTree(root)
    tree.write(str(out), encoding="utf-8", xml_declaration=True, pretty_print=True)
    return out


def export_explicit_only_gold_xml(
    canonical_docs: Sequence[CanonicalDocument],
    out_xml_path: str | Path,
) -> Path:
    """Export gold XML containing ONLY explicit opinions (`target != 'NULL' and to > 0`) for clean Task D OTE evaluation."""
    root = ET.Element("Documents")
    for doc in canonical_docs:
        doc_elem = ET.SubElement(root, "Document", attrib={"id": doc.doc_id})
        explicit_ops = [
            op for op in doc.opinions if (not op.is_implicit and op.target_xml != "NULL" and op.end_xml > 0)
        ]
        if explicit_ops:
            ops_elem = ET.SubElement(doc_elem, "Opinions")
            for op in explicit_ops:
                ET.SubElement(ops_elem, "Opinion", attrib=op.xml_attributes)
        rel_elem = ET.SubElement(doc_elem, "relevance")
        rel_elem.text = "true" if doc.relevance else "false"
        sent_elem = ET.SubElement(doc_elem, "sentiment")
        sent_elem.text = doc.document_sentiment or "neutral"
        text_elem = ET.SubElement(doc_elem, "text")
        text_elem.text = doc.text

    out = Path(out_xml_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    tree = ET.ElementTree(root)
    tree.write(str(out), encoding="utf-8", xml_declaration=True, pretty_print=True)
    return out


def run_official_java_evaluator(
    subtask: str,
    predicted_file: str | Path,
    gold_file: str | Path,
    jar_path: str | Path = DEFAULT_JAR_PATH,
) -> dict[str, Any]:
    """Invoke `java -jar EvaluationScript.jar <subtask> <PREDICTED_FILE> <GOLD_FILE>` and parse its output.

    Note: The first file argument is PREDICTED and the second is GOLD, matching `Evaluate.java:51-53`.
    """
    require_jaxb = str(predicted_file).endswith(".xml") or str(gold_file).endswith(".xml")
    java_bin = find_java_binary(require_jaxb=require_jaxb)

    cmd = [
        java_bin,
        "-jar",
        str(jar_path),
        subtask,
        str(predicted_file),
        str(gold_file),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(
            f"EvaluationScript.jar failed (exit {proc.returncode}):\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
        )

    f1_matches = _MICRO_F1_RE.findall(proc.stdout)
    inst_matches = _INSTANCES_RE.findall(proc.stdout)

    def _to_float(val: str) -> float:
        return 0.0 if val == "NaN" else float(val)

    result: dict[str, Any] = {
        "subtask": subtask,
        "predicted_file": str(predicted_file),
        "gold_file": str(gold_file),
        "raw_stdout": proc.stdout,
    }

    if subtask == "category" and len(f1_matches) >= 2:
        result["task_c1_category_micro_f1"] = _to_float(f1_matches[0])
        result["task_c1_instances"] = int(inst_matches[0]) if len(inst_matches) >= 1 else 0
        result["task_c2_category_sentiment_micro_f1"] = _to_float(f1_matches[1])
        result["task_c2_instances"] = int(inst_matches[1]) if len(inst_matches) >= 2 else 0
    elif subtask == "OTE" and len(f1_matches) >= 2:
        result["ote_exact_micro_f1"] = _to_float(f1_matches[0])
        result["ote_exact_instances"] = int(inst_matches[0]) if len(inst_matches) >= 1 else 0
        result["ote_overlap_micro_f1"] = _to_float(f1_matches[1])
        result["ote_overlap_instances"] = int(inst_matches[1]) if len(inst_matches) >= 2 else 0
    elif f1_matches:
        result["micro_f1"] = _to_float(f1_matches[0])

    return result
