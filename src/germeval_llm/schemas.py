"""Typed Pydantic schemas for canonical GermEval 2017 documents, tasks, and evaluations."""

from __future__ import annotations

from typing import Any, Literal
from pydantic import BaseModel, ConfigDict, Field


class CanonicalOpinion(BaseModel):
    """Single XML <Opinion> record preserved verbatim from the original GermEval 2017 XML."""

    model_config = ConfigDict(extra="forbid")

    category_full: str = Field(..., description="Full XML category attribute, e.g. 'Zugfahrt#Haupt'")
    category: str = Field(..., description="Main category before '#', e.g. 'Zugfahrt'")
    subcategory: str = Field(..., description="Subcategory after '#' if present, else ''")
    polarity: str = Field(..., description="Original XML polarity attribute, e.g. 'negative'")
    target_xml: str = Field(..., description="Original XML target attribute ('NULL' for implicit)")
    start_xml: int = Field(..., description="Integer value of XML 'from' attribute")
    end_xml: int = Field(..., description="Integer value of XML 'to' attribute")
    start2_xml: int = Field(0, description="Integer value of XML 'from2' attribute if present, else 0")
    end2_xml: int = Field(0, description="Integer value of XML 'to2' attribute if present, else 0")
    is_implicit: bool = Field(..., description="True when target_xml == 'NULL' and start_xml == 0 and end_xml == 0")
    is_discontinuous_star: bool = Field(..., description="True when '*' in target_xml for an explicit target")
    xml_attributes: dict[str, str] = Field(
        default_factory=dict,
        description="Exact original XML attribute dictionary for lossless round-trip verification",
    )


class CanonicalDocument(BaseModel):
    """Single XML <Document> record preserved verbatim with all opinions and metadata."""

    model_config = ConfigDict(extra="forbid")

    doc_id: str = Field(..., description="Document @id attribute from XML")
    source_split: str = Field(..., description="Split identifier: train, dev, test_syn, test_dia")
    source_ordinal: int = Field(..., description="0-based index of the document in the source XML file")
    text: str = Field(..., description="Exact document <text> string without normalization")
    relevance: bool = Field(..., description="Boolean parsed from <relevance> ('true'/'false')")
    document_sentiment: str = Field(..., description="Document-level <sentiment> string")
    has_opinions_element: bool = Field(
        True, description="Whether the XML <Document> contained an <Opinions> child tag"
    )
    opinions: list[CanonicalOpinion] = Field(default_factory=list)


class ValidationIssue(BaseModel):
    """Single validation or audit finding on a canonical document/opinion."""

    model_config = ConfigDict(extra="forbid")

    severity: Literal["INFO", "WARNING", "ERROR"]
    split: str
    doc_id: str
    source_ordinal: int
    opinion_index: int | None = None
    issue_code: str
    message: str
    details: dict[str, Any] = Field(default_factory=dict)


class ChatMessage(BaseModel):
    """Single conversational turn for instruction tuning or prompting."""

    model_config = ConfigDict(extra="forbid")

    role: Literal["user", "assistant", "model", "system"]
    content: str


class TaskCExample(BaseModel):
    """Derived Task C instruction-tuning and prompting example (no offsets)."""

    model_config = ConfigDict(extra="forbid")

    example_id: str
    doc_id: str
    source_split: str
    source_ordinal: int
    text: str
    relevance: bool
    gold_labels: list[str] = Field(
        ..., description="Sorted unique 'Category:polarity' labels for this document (Task C2)"
    )
    gold_categories: list[str] = Field(
        ..., description="Sorted unique main 'Category' labels for this document (Task C1)"
    )
    prompt: list[ChatMessage]
    completion: list[ChatMessage]


class TaskDTargetItem(BaseModel):
    """Model-facing Task D target extraction item (strictly NO character offsets or token indices)."""

    model_config = ConfigDict(extra="forbid")

    target: str = Field(..., description="Verbatim copied target string (or ' * '-joined discontinuous parts)")
    category: str = Field(..., description="Main aspect category")
    polarity: str = Field(..., description="Aspect polarity: positive, neutral, negative")


class TaskDExample(BaseModel):
    """Derived Task D instruction-tuning and prompting example (strictly NO offsets)."""

    model_config = ConfigDict(extra="forbid")

    example_id: str
    doc_id: str
    source_split: str
    source_ordinal: int
    text: str
    relevance: bool
    view: Literal["exact_and_star", "contiguous_only"] = "exact_and_star"
    gold_targets: list[TaskDTargetItem]
    prompt: list[ChatMessage]
    completion: list[ChatMessage]


class AlignedTargetSpan(BaseModel):
    """Post-processed Task D prediction aligned deterministically to character offsets in source text."""

    model_config = ConfigDict(extra="forbid")

    target: str
    category: str
    polarity: str
    alignment_status: Literal[
        "unique_contiguous",
        "ambiguous_contiguous",
        "discontinuous_star",
        "unaligned",
    ]
    start_char: int | None = None
    end_char: int | None = None
    occurrence_count: int = 0
    assigned_occurrence_index: int | None = None


class PredictionRecord(BaseModel):
    """Saved per-document model prediction record for Task C or Task D."""

    model_config = ConfigDict(extra="forbid")

    run_id: str
    task: Literal["C", "D", "CD"]
    split: str
    doc_id: str
    source_ordinal: int
    raw_output: str
    parse_status: Literal[
        "VALID_EXACT",
        "VALID_FENCED",
        "INVALID_JSON",
        "INVALID_SCHEMA",
        "UNKNOWN_LABEL",
    ]
    parse_errors: list[str] = Field(default_factory=list)
    parsed_labels: list[str] = Field(default_factory=list, description="For Task C: sorted 'Category:polarity'")
    parsed_targets: list[TaskDTargetItem] = Field(default_factory=list, description="For Task D: extracted targets")
    aligned_spans: list[AlignedTargetSpan] = Field(
        default_factory=list, description="For Task D: deterministically aligned character spans"
    )
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: float = 0.0
