"""Validation, anomaly auditing, and dataset statistics for canonical GermEval 2017 documents."""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
from typing import Any

from germeval_llm.schemas import CanonicalDocument, ValidationIssue
from germeval_llm.taxonomy import (
    KNOWN_CATEGORY_MAIN_TYPOS,
    KNOWN_POLARITY_TYPOS,
    MAIN_CATEGORIES_19,
    POLARITIES,
    RARE_UNSEEN_CATEGORIES,
)


def check_discontinuous_parts_in_text(text: str, target_xml: str) -> bool:
    """Check whether all '*'-separated parts of a discontinuous target appear in order in text."""
    parts = [p.strip() for p in target_xml.split("*") if p.strip()]
    if not parts:
        return False
    pos = 0
    for part in parts:
        idx = text.find(part, pos)
        if idx == -1:
            return False
        pos = idx + len(part)
    return True


def validate_canonical_document(doc: CanonicalDocument) -> list[ValidationIssue]:
    """Audit a single CanonicalDocument and return all discovered ValidationIssues."""
    issues: list[ValidationIssue] = []

    if not doc.text.strip():
        issues.append(
            ValidationIssue(
                severity="ERROR",
                split=doc.source_split,
                doc_id=doc.doc_id,
                source_ordinal=doc.source_ordinal,
                issue_code="EMPTY_DOCUMENT_TEXT",
                message="Document text is empty or whitespace-only.",
            )
        )

    if doc.relevance and len(doc.opinions) == 0:
        issues.append(
            ValidationIssue(
                severity="INFO",
                split=doc.source_split,
                doc_id=doc.doc_id,
                source_ordinal=doc.source_ordinal,
                issue_code="RELEVANT_DOC_WITHOUT_OPINIONS",
                message="Document has relevance=true but zero <Opinion> elements.",
            )
        )

    if (not doc.relevance) and len(doc.opinions) > 0:
        issues.append(
            ValidationIssue(
                severity="WARNING",
                split=doc.source_split,
                doc_id=doc.doc_id,
                source_ordinal=doc.source_ordinal,
                issue_code="IRRELEVANT_DOC_WITH_OPINIONS",
                message=f"Document has relevance=false but {len(doc.opinions)} opinions.",
            )
        )

    for op_idx, op in enumerate(doc.opinions):
        # 1. Category checks
        if op.category in KNOWN_CATEGORY_MAIN_TYPOS:
            issues.append(
                ValidationIssue(
                    severity="WARNING",
                    split=doc.source_split,
                    doc_id=doc.doc_id,
                    source_ordinal=doc.source_ordinal,
                    opinion_index=op_idx,
                    issue_code="CATEGORY_TYPO",
                    message=f"Known category typo {op.category_full!r} -> {KNOWN_CATEGORY_MAIN_TYPOS[op.category]!r}",
                    details={"category_full": op.category_full, "category": op.category},
                )
            )
        elif op.category in RARE_UNSEEN_CATEGORIES:
            issues.append(
                ValidationIssue(
                    severity="WARNING",
                    split=doc.source_split,
                    doc_id=doc.doc_id,
                    source_ordinal=doc.source_ordinal,
                    opinion_index=op_idx,
                    issue_code="UNSEEN_IN_TRAIN_CATEGORY",
                    message=f"Category {op.category!r} does not appear in train_v1.4.xml.",
                    details={"category_full": op.category_full},
                )
            )
        elif op.category not in MAIN_CATEGORIES_19:
            issues.append(
                ValidationIssue(
                    severity="ERROR",
                    split=doc.source_split,
                    doc_id=doc.doc_id,
                    source_ordinal=doc.source_ordinal,
                    opinion_index=op_idx,
                    issue_code="UNKNOWN_CATEGORY",
                    message=f"Unknown main category {op.category!r}.",
                    details={"category_full": op.category_full},
                )
            )

        # 2. Polarity checks
        if op.polarity in KNOWN_POLARITY_TYPOS:
            issues.append(
                ValidationIssue(
                    severity="WARNING",
                    split=doc.source_split,
                    doc_id=doc.doc_id,
                    source_ordinal=doc.source_ordinal,
                    opinion_index=op_idx,
                    issue_code="POLARITY_TYPO",
                    message=f"Known polarity typo {op.polarity!r} -> {KNOWN_POLARITY_TYPOS[op.polarity]!r}",
                    details={"polarity": op.polarity},
                )
            )
        elif op.polarity not in POLARITIES:
            issues.append(
                ValidationIssue(
                    severity="ERROR",
                    split=doc.source_split,
                    doc_id=doc.doc_id,
                    source_ordinal=doc.source_ordinal,
                    opinion_index=op_idx,
                    issue_code="UNKNOWN_POLARITY",
                    message=f"Unknown polarity {op.polarity!r}.",
                    details={"polarity": op.polarity},
                )
            )

        # 3. Target & offset consistency checks
        if op.is_implicit:
            continue

        # Check for NULL vs (0,0) inconsistency (e.g., from > 0, to == 0 in test_TIMESTAMP2)
        if (op.target_xml == "NULL") != (op.end_xml == 0):
            issues.append(
                ValidationIssue(
                    severity="WARNING",
                    split=doc.source_split,
                    doc_id=doc.doc_id,
                    source_ordinal=doc.source_ordinal,
                    opinion_index=op_idx,
                    issue_code="NULL_OFFSET_ANOMALY",
                    message=f"Inconsistent NULL/zero offset: target={op.target_xml!r}, from={op.start_xml}, to={op.end_xml}",
                    details={
                        "target_xml": op.target_xml,
                        "start_xml": op.start_xml,
                        "end_xml": op.end_xml,
                    },
                )
            )
            continue

        slice_text = doc.text[op.start_xml : op.end_xml]
        if slice_text != op.target_xml:
            if op.is_discontinuous_star:
                parts_ok = check_discontinuous_parts_in_text(doc.text, op.target_xml)
                issues.append(
                    ValidationIssue(
                        severity="INFO" if parts_ok else "WARNING",
                        split=doc.source_split,
                        doc_id=doc.doc_id,
                        source_ordinal=doc.source_ordinal,
                        opinion_index=op_idx,
                        issue_code="DISCONTINUOUS_STAR_TARGET",
                        message=f"Discontinuous '*' target {op.target_xml!r} (parts_in_order={parts_ok})",
                        details={
                            "target_xml": op.target_xml,
                            "slice_text": slice_text,
                            "start_xml": op.start_xml,
                            "end_xml": op.end_xml,
                            "start2_xml": op.start2_xml,
                            "end2_xml": op.end2_xml,
                            "parts_in_order": parts_ok,
                        },
                    )
                )
            else:
                issues.append(
                    ValidationIssue(
                        severity="WARNING",
                        split=doc.source_split,
                        doc_id=doc.doc_id,
                        source_ordinal=doc.source_ordinal,
                        opinion_index=op_idx,
                        issue_code="NON_STAR_OFFSET_MISMATCH",
                        message=f"Non-'*' target mismatch: slice={slice_text!r} != target={op.target_xml!r}",
                        details={
                            "target_xml": op.target_xml,
                            "slice_text": slice_text,
                            "start_xml": op.start_xml,
                            "end_xml": op.end_xml,
                            "target_in_text": op.target_xml in doc.text,
                        },
                    )
                )

    return issues


def audit_splits(
    split_docs: dict[str, list[CanonicalDocument]],
) -> tuple[list[ValidationIssue], dict[str, Any]]:
    """Run validation across all splits and compute summary statistics and cross-split duplicate audits."""
    all_issues: list[ValidationIssue] = []
    split_stats: dict[str, Any] = {}

    text_to_splits: dict[str, list[tuple[str, str]]] = {}
    id_to_splits: dict[str, list[str]] = {}

    for split_name, docs in split_docs.items():
        cat_counter: Counter[str] = Counter()
        pol_counter: Counter[str] = Counter()
        cat_pol_counter: Counter[str] = Counter()
        issue_counter: Counter[str] = Counter()

        n_docs = len(docs)
        n_rel_true = sum(1 for d in docs if d.relevance)
        n_rel_false = n_docs - n_rel_true
        n_zero_ops = sum(1 for d in docs if len(d.opinions) == 0)
        n_total_ops = sum(len(d.opinions) for d in docs)
        n_implicit = 0
        n_explicit = 0
        n_exact_contiguous = 0
        n_discontinuous_star = 0
        n_non_star_mismatch = 0
        n_repeated_in_doc = 0
        n_from2_nonzero = 0

        for doc in docs:
            text_to_splits.setdefault(doc.text, []).append((split_name, doc.doc_id))
            id_to_splits.setdefault(doc.doc_id, []).append(split_name)

            doc_issues = validate_canonical_document(doc)
            all_issues.extend(doc_issues)
            for iss in doc_issues:
                issue_counter[iss.issue_code] += 1

            for op in doc.opinions:
                cat_counter[op.category] += 1
                pol_counter[op.polarity] += 1
                cat_pol_counter[f"{op.category}:{op.polarity}"] += 1
                if op.start2_xml != 0 or op.end2_xml != 0:
                    n_from2_nonzero += 1
                if op.is_implicit:
                    n_implicit += 1
                else:
                    n_explicit += 1
                    if doc.text[op.start_xml : op.end_xml] == op.target_xml:
                        n_exact_contiguous += 1
                    elif op.is_discontinuous_star:
                        n_discontinuous_star += 1
                    else:
                        n_non_star_mismatch += 1
                    if op.target_xml != "NULL" and doc.text.count(op.target_xml) > 1:
                        n_repeated_in_doc += 1

        split_stats[split_name] = {
            "num_documents": n_docs,
            "num_unique_doc_ids": len({d.doc_id for d in docs}),
            "num_relevance_true": n_rel_true,
            "num_relevance_false": n_rel_false,
            "num_docs_zero_opinions": n_zero_ops,
            "num_opinions_total": n_total_ops,
            "num_opinions_implicit": n_implicit,
            "num_opinions_explicit": n_explicit,
            "num_explicit_exact_contiguous": n_exact_contiguous,
            "num_explicit_discontinuous_star": n_discontinuous_star,
            "num_explicit_non_star_mismatch": n_non_star_mismatch,
            "num_explicit_repeated_target_in_doc": n_repeated_in_doc,
            "num_opinions_from2_nonzero": n_from2_nonzero,
            "category_counts": dict(sorted(cat_counter.items())),
            "polarity_counts": dict(sorted(pol_counter.items())),
            "category_polarity_counts": dict(sorted(cat_pol_counter.items())),
            "issue_counts": dict(sorted(issue_counter.items())),
        }

    # Cross-split exact text and doc_id duplicate audit
    cross_split_text_dups = {
        txt: locs
        for txt, locs in text_to_splits.items()
        if len({s for s, _ in locs}) > 1
    }
    cross_split_id_dups = {
        did: sps for did, sps in id_to_splits.items() if len(set(sps)) > 1
    }

    summary = {
        "splits": split_stats,
        "cross_split_audit": {
            "exact_text_duplicates_across_splits_count": len(cross_split_text_dups),
            "doc_id_duplicates_across_splits_count": len(cross_split_id_dups),
        },
        "total_issues": len(all_issues),
    }
    return all_issues, summary


def save_validation_audit(
    issues: list[ValidationIssue],
    summary: dict[str, Any],
    audit_path: str | Path,
    stats_path: str | Path,
) -> None:
    """Save machine-readable validation_audit.json and statistics.json."""
    apath = Path(audit_path)
    apath.parent.mkdir(parents=True, exist_ok=True)
    apath.write_text(
        json.dumps(
            {
                "total_issues": len(issues),
                "issues": [iss.model_dump() for iss in issues],
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    spath = Path(stats_path)
    spath.parent.mkdir(parents=True, exist_ok=True)
    spath.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
