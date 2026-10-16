"""Unit tests for XML parsing, canonical representation, round-trip verification, and validation auditing."""

from __future__ import annotations

from pathlib import Path
import lxml.etree as ET

from germeval_llm.validate import validate_canonical_document
from germeval_llm.xml_parser import (
    canonical_to_xml_tree,
    parse_document_element,
    parse_xml_file,
    verify_lossless_roundtrip,
)


def test_parse_document_variants_and_roundtrip(tmp_path: Path) -> None:
    xml_content = """<?xml version="1.0" encoding="UTF-8"?>
<Documents>
  <Document id="doc-irrelevant">
    <Opinions/>
    <relevance>false</relevance>
    <sentiment>neutral</sentiment>
    <text>Studie: Messen bringen Berlin Millionen.</text>
  </Document>
  <Document id="doc-implicit-and-explicit">
    <Opinions>
      <Opinion category="Allgemein#Haupt" from="0" to="0" target="NULL" polarity="neutral"/>
      <Opinion category="Zugfahrt#Haupt" from="4" to="7" target="Zug" polarity="negative"/>
      <Opinion category="Zugfahrt#Pünktlichkeit" from="4" to="7" target="Zug" polarity="positive"/>
      <Opinion category="Informationen#Haupt" from="19" to="44" from2="0" to2="0" target="zeigt * an" polarity="negative"/>
    </Opinions>
    <relevance>true</relevance>
    <sentiment>negative</sentiment>
    <text>Der Zug fällt aus, zeigt falsche Zeiten an.</text>
  </Document>
</Documents>
"""
    xml_file = tmp_path / "sample.xml"
    xml_file.write_text(xml_content, encoding="utf-8")

    docs = parse_xml_file(xml_file, source_split="train")
    assert len(docs) == 2

    d0 = docs[0]
    assert d0.doc_id == "doc-irrelevant"
    assert d0.relevance is False
    assert len(d0.opinions) == 0

    d1 = docs[1]
    assert d1.doc_id == "doc-implicit-and-explicit"
    assert d1.relevance is True
    assert len(d1.opinions) == 4
    assert d1.opinions[0].is_implicit is True
    assert d1.opinions[1].is_implicit is False
    assert d1.opinions[1].category == "Zugfahrt"
    assert d1.opinions[1].subcategory == "Haupt"
    assert d1.opinions[3].is_discontinuous_star is True
    assert "from2" in d1.opinions[3].xml_attributes

    ok, errors = verify_lossless_roundtrip(xml_file, docs)
    assert ok is True, f"Roundtrip failed: {errors}"

    reconstructed_tree = canonical_to_xml_tree(docs)
    reconstructed_docs = reconstructed_tree.getroot().findall("Document")
    assert len(reconstructed_docs) == 2


def test_validate_canonical_detects_typos_and_anomalies() -> None:
    xml_str = """
    <Document id="doc-typos">
      <Opinions>
        <Opinion category="Sichaupt" from="0" to="4" target="Bahn" polarity="positve"/>
        <Opinion category="QR-Code#Haupt" from="0" to="4" target="Bahn" polarity=" negative"/>
        <Opinion category="DB_App_und_Website#Haupt" from="68" to="0" target="#Navigator" polarity="neutral"/>
      </Opinions>
      <relevance>true</relevance>
      <sentiment>neutral</sentiment>
      <text>Bahn fährt heute pünktlich.</text>
    </Document>
    """
    elem = ET.fromstring(xml_str)
    doc = parse_document_element(elem, source_split="train", source_ordinal=0)
    issues = validate_canonical_document(doc)
    codes = {iss.issue_code for iss in issues}

    assert "CATEGORY_TYPO" in codes
    assert "POLARITY_TYPO" in codes
    assert "UNSEEN_IN_TRAIN_CATEGORY" in codes
    assert "NULL_OFFSET_ANOMALY" in codes
