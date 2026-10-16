"""Unit tests for Task C/D conversion, typo patching, few-shot selection, and learning-curve sampling."""

from __future__ import annotations

import json
import lxml.etree as ET

from germeval_llm.convert import (
    derive_task_c_example,
    derive_task_d_example,
    extract_task_c_label_sets,
    extract_task_d_targets,
    sample_learning_curve_subset,
    select_few_shot_examples_task_c,
)
from germeval_llm.xml_parser import parse_document_element


def _make_doc(xml_str: str, split: str = "train", ordinal: int = 0):
    return parse_document_element(ET.fromstring(xml_str), source_split=split, source_ordinal=ordinal)


def test_task_c_label_deduplication_and_contradictory_polarities() -> None:
    doc = _make_doc(
        """
        <Document id="d1">
          <Opinions>
            <Opinion category="Zugfahrt#Haupt" from="4" to="7" target="Zug" polarity="negative"/>
            <Opinion category="Zugfahrt#Pünktlichkeit" from="4" to="7" target="Zug" polarity="negative"/>
            <Opinion category="Zugfahrt#Haupt" from="0" to="0" target="NULL" polarity="positive"/>
            <Opinion category="Allgemein#Haupt" from="0" to="0" target="NULL" polarity="neutral"/>
          </Opinions>
          <relevance>true</relevance>
          <sentiment>negative</sentiment>
          <text>Der Zug war erst spät, aber insgesamt gut.</text>
        </Document>
        """
    )
    labels_c2, cats_c1 = extract_task_c_label_sets(doc)
    assert labels_c2 == [
        "Allgemein:neutral",
        "Zugfahrt:positive",
        "Zugfahrt:negative",
    ]
    assert cats_c1 == ["Allgemein", "Zugfahrt"]


def test_typo_patching_applied_only_when_enabled() -> None:
    doc = _make_doc(
        """
        <Document id="d2">
          <Opinions>
            <Opinion category="Sichaupt" from="0" to="4" target="Bahn" polarity="positve"/>
          </Opinions>
          <relevance>true</relevance>
          <sentiment>positive</sentiment>
          <text>Bahn ist sicher.</text>
        </Document>
        """
    )
    raw_labels, _ = extract_task_c_label_sets(doc, apply_typo_patches=False)
    assert raw_labels == ["Sichaupt:positve"]

    patched_labels, patched_cats = extract_task_c_label_sets(doc, apply_typo_patches=True)
    assert patched_labels == ["Sicherheit:positive"]
    assert patched_cats == ["Sicherheit"]


def test_task_d_conversion_excludes_implicit_and_contains_no_offsets() -> None:
    doc = _make_doc(
        """
        <Document id="d3">
          <Opinions>
            <Opinion category="Allgemein#Haupt" from="0" to="0" target="NULL" polarity="neutral"/>
            <Opinion category="Zugfahrt#Haupt" from="4" to="7" target="Zug" polarity="negative"/>
            <Opinion category="Informationen#Haupt" from="19" to="44" target="zeigt * an" polarity="negative"/>
          </Opinions>
          <relevance>true</relevance>
          <sentiment>negative</sentiment>
          <text>Der Zug fällt aus, zeigt falsche Zeiten an.</text>
        </Document>
        """
    )
    ex_star = derive_task_d_example(doc, view="exact_and_star")
    assert len(ex_star.gold_targets) == 2
    assert ex_star.gold_targets[0].target == "Zug"
    assert ex_star.gold_targets[1].target == "zeigt * an"

    # Verify completion JSON contains no offset keys
    comp_json = json.loads(ex_star.completion[0].content)
    assert "targets" in comp_json
    for item in comp_json["targets"]:
        assert set(item.keys()) == {"target", "category", "polarity"}

    # Contiguous-only view excludes '*' target
    targets_contig = extract_task_d_targets(doc, view="contiguous_only")
    assert len(targets_contig) == 1
    assert targets_contig[0].target == "Zug"


def test_few_shot_and_learning_curve_sampling_determinism() -> None:
    exs = []
    for i in range(50):
        cat = "Allgemein" if i % 2 == 0 else ("Toiletten" if i % 7 == 0 else "Zugfahrt")
        ops_xml = (
            ""
            if i % 5 == 0
            else f'<Opinion category="{cat}#Haupt" from="0" to="0" target="NULL" polarity="negative"/>'
        )
        doc = _make_doc(
            f"""
            <Document id="doc-{i}">
              <Opinions>{ops_xml}</Opinions>
              <relevance>{'false' if i % 5 == 0 else 'true'}</relevance>
              <sentiment>negative</sentiment>
              <text>Beispieltext Nummer {i} über die Deutsche Bahn im Fernverkehr und Regionalverkehr.</text>
            </Document>
            """,
            ordinal=i,
        )
        exs.append(derive_task_c_example(doc))

    shots_a = select_few_shot_examples_task_c(exs, n_shots=5, strategy="stratified_diverse", seed=42)
    shots_b = select_few_shot_examples_task_c(exs, n_shots=5, strategy="stratified_diverse", seed=42)
    assert [e.example_id for e in shots_a] == [e.example_id for e in shots_b]
    assert any(len(e.gold_labels) == 0 for e in shots_a)

    sub_strat = sample_learning_curve_subset(exs, subset_size=15, strategy="stratified", seed=42)
    assert len(sub_strat) == 15
    assert any("Toiletten:negative" in e.gold_labels for e in sub_strat)
