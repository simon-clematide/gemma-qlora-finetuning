"""Lossless XML parser and round-trip converter for GermEval 2017 ABSA datasets."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable
import lxml.etree as ET

from germeval_llm.schemas import CanonicalDocument, CanonicalOpinion


def compute_sha256(path: str | Path) -> str:
    """Compute hex SHA-256 digest of a file."""
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_opinion_element(op_elem: ET._Element) -> CanonicalOpinion:
    """Parse a single <Opinion> element into a CanonicalOpinion without altering values."""
    attrs = {str(k): str(v) for k, v in op_elem.attrib.items()}
    category_full = attrs.get("category", "")
    if "#" in category_full:
        category, subcategory = category_full.split("#", 1)
    else:
        category, subcategory = category_full, ""

    polarity = attrs.get("polarity", "")
    target_xml = attrs.get("target", "")
    start_xml = int(attrs.get("from", "0"))
    end_xml = int(attrs.get("to", "0"))
    start2_xml = int(attrs.get("from2", "0"))
    end2_xml = int(attrs.get("to2", "0"))

    is_implicit = (target_xml == "NULL") and (start_xml == 0) and (end_xml == 0)
    is_discontinuous_star = (not is_implicit) and ("*" in target_xml)

    return CanonicalOpinion(
        category_full=category_full,
        category=category,
        subcategory=subcategory,
        polarity=polarity,
        target_xml=target_xml,
        start_xml=start_xml,
        end_xml=end_xml,
        start2_xml=start2_xml,
        end2_xml=end2_xml,
        is_implicit=is_implicit,
        is_discontinuous_star=is_discontinuous_star,
        xml_attributes=attrs,
    )


def parse_document_element(
    doc_elem: ET._Element,
    source_split: str,
    source_ordinal: int,
) -> CanonicalDocument:
    """Parse a single <Document> element into a CanonicalDocument."""
    doc_id = str(doc_elem.get("id", ""))
    text_elem = doc_elem.find("text")
    text = text_elem.text if (text_elem is not None and text_elem.text is not None) else ""

    rel_elem = doc_elem.find("relevance")
    rel_str = (rel_elem.text or "").strip().lower() if rel_elem is not None else "false"
    relevance = rel_str == "true"

    sent_elem = doc_elem.find("sentiment")
    document_sentiment = (sent_elem.text or "") if sent_elem is not None else ""

    opinions_container = doc_elem.find("Opinions")
    has_opinions_element = opinions_container is not None
    opinions: list[CanonicalOpinion] = []
    if opinions_container is not None:
        for op_elem in opinions_container.findall("Opinion"):
            opinions.append(parse_opinion_element(op_elem))

    return CanonicalDocument(
        doc_id=doc_id,
        source_split=source_split,
        source_ordinal=source_ordinal,
        text=text,
        relevance=relevance,
        document_sentiment=document_sentiment,
        has_opinions_element=has_opinions_element,
        opinions=opinions,
    )


def parse_xml_file(xml_path: str | Path, source_split: str) -> list[CanonicalDocument]:
    """Parse an entire GermEval 2017 XML file into a list of CanonicalDocuments."""
    parser = ET.XMLParser(remove_blank_text=False, resolve_entities=True, encoding="utf-8")
    tree = ET.parse(str(xml_path), parser=parser)
    root = tree.getroot()
    doc_elems = root.findall("Document")
    return [
        parse_document_element(doc_elem, source_split=source_split, source_ordinal=idx)
        for idx, doc_elem in enumerate(doc_elems)
    ]


def canonical_to_xml_tree(documents: Iterable[CanonicalDocument]) -> ET._ElementTree:
    """Reconstruct an XML ElementTree from canonical documents for lossless round-trip verification."""
    root = ET.Element("Documents")
    for doc in documents:
        doc_elem = ET.SubElement(root, "Document", attrib={"id": doc.doc_id})
        if doc.has_opinions_element or doc.opinions:
            ops_elem = ET.SubElement(doc_elem, "Opinions")
            for op in doc.opinions:
                if op.xml_attributes:
                    ET.SubElement(ops_elem, "Opinion", attrib=op.xml_attributes)
                else:
                    attrib = {
                        "category": op.category_full,
                        "from": str(op.start_xml),
                        "to": str(op.end_xml),
                        "target": op.target_xml,
                        "polarity": op.polarity,
                    }
                    if op.start2_xml or op.end2_xml:
                        attrib["from2"] = str(op.start2_xml)
                        attrib["to2"] = str(op.end2_xml)
                    ET.SubElement(ops_elem, "Opinion", attrib=attrib)
        rel_elem = ET.SubElement(doc_elem, "relevance")
        rel_elem.text = "true" if doc.relevance else "false"
        sent_elem = ET.SubElement(doc_elem, "sentiment")
        sent_elem.text = doc.document_sentiment
        text_elem = ET.SubElement(doc_elem, "text")
        text_elem.text = doc.text
    return ET.ElementTree(root)


def verify_lossless_roundtrip(xml_path: str | Path, documents: list[CanonicalDocument]) -> tuple[bool, list[str]]:
    """Verify that parsed CanonicalDocuments match the source XML structure and attributes 1-to-1."""
    parser = ET.XMLParser(remove_blank_text=False, resolve_entities=True, encoding="utf-8")
    tree = ET.parse(str(xml_path), parser=parser)
    src_docs = tree.getroot().findall("Document")

    errors: list[str] = []
    if len(src_docs) != len(documents):
        errors.append(f"Document count mismatch: XML={len(src_docs)} vs Canonical={len(documents)}")
        return False, errors

    for idx, (src_doc, can_doc) in enumerate(zip(src_docs, documents)):
        if (src_doc.get("id") or "") != can_doc.doc_id:
            errors.append(f"Doc {idx}: id mismatch {src_doc.get('id')!r} != {can_doc.doc_id!r}")
        src_text = src_doc.findtext("text") or ""
        if src_text != can_doc.text:
            errors.append(f"Doc {idx} ({can_doc.doc_id}): text mismatch")
        src_rel = (src_doc.findtext("relevance") or "").strip().lower() == "true"
        if src_rel != can_doc.relevance:
            errors.append(f"Doc {idx} ({can_doc.doc_id}): relevance mismatch")
        src_sent = src_doc.findtext("sentiment") or ""
        if src_sent != can_doc.document_sentiment:
            errors.append(f"Doc {idx} ({can_doc.doc_id}): sentiment mismatch")

        src_ops = src_doc.findall("Opinions/Opinion")
        if len(src_ops) != len(can_doc.opinions):
            errors.append(
                f"Doc {idx} ({can_doc.doc_id}): opinion count mismatch {len(src_ops)} != {len(can_doc.opinions)}"
            )
            continue
        for op_idx, (src_op, can_op) in enumerate(zip(src_ops, can_doc.opinions)):
            src_attrs = {str(k): str(v) for k, v in src_op.attrib.items()}
            if src_attrs != can_op.xml_attributes:
                errors.append(
                    f"Doc {idx} op {op_idx}: attribute dict mismatch {src_attrs} != {can_op.xml_attributes}"
                )

    return len(errors) == 0, errors


def save_canonical_jsonl(
    documents: Iterable[CanonicalDocument],
    out_path: str | Path,
    write_gzip: bool = True,
) -> Path:
    """Save canonical documents to a UTF-8 JSONL file (and a compact `.jsonl.gz` mirror for Git/Colab)."""
    import gzip

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(doc.model_dump(), ensure_ascii=False) + "\n" for doc in documents]
    payload = "".join(lines)
    out.write_text(payload, encoding="utf-8")
    if write_gzip and not str(out).endswith(".gz"):
        gz_path = Path(f"{out}.gz")
        with gzip.open(gz_path, "wt", encoding="utf-8", compresslevel=9) as gf:
            gf.write(payload)
    return out


def load_canonical_jsonl(jsonl_path: str | Path) -> list[CanonicalDocument]:
    """Load canonical documents from a UTF-8 `.jsonl` or `.jsonl.gz` file.

    Automatically falls back to `<jsonl_path>.gz` (and decompresses it to `<jsonl_path>` if writable)
    when the uncompressed `.jsonl` is absent (e.g., on a fresh Git clone in Google Colab).
    """
    import gzip

    p = Path(jsonl_path)
    if not p.exists():
        gz_candidate = Path(f"{p}.gz")
        if gz_candidate.exists():
            with gzip.open(gz_candidate, "rt", encoding="utf-8") as gf:
                content = gf.read()
            try:
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(content, encoding="utf-8")
            except OSError:
                pass
            return [
                CanonicalDocument.model_validate_json(line.strip())
                for line in content.splitlines()
                if line.strip()
            ]
        raise FileNotFoundError(f"Neither {p} nor {gz_candidate} found.")

    open_fn = gzip.open if str(p).endswith(".gz") else open
    docs: list[CanonicalDocument] = []
    with open_fn(p, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                docs.append(CanonicalDocument.model_validate_json(line))
    return docs

