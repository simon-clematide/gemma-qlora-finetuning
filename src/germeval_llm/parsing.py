"""Shared strict output parser and validator for Task C and Task D model completions."""

from __future__ import annotations

import json
import re
from typing import Literal

from germeval_llm.schemas import TaskDTargetItem
from germeval_llm.taxonomy import (
    MAIN_CATEGORIES_19,
    POLARITIES,
    get_allowed_task_c_labels,
    sort_task_c_labels,
)

ParseStatus = Literal[
    "VALID_EXACT",
    "VALID_FENCED",
    "INVALID_JSON",
    "INVALID_SCHEMA",
    "UNKNOWN_LABEL",
]

_FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)
_BRACE_RE = re.compile(r"(\{.*\})", re.DOTALL)


def _extract_json_object(raw_text: str) -> tuple[dict | None, bool, str | None]:
    """Attempt to parse a JSON object from raw model text.

    Returns (parsed_dict, used_fence_or_extraction, error_message).
    """
    stripped = raw_text.strip()
    if not stripped:
        return None, False, "Empty model output"

    try:
        obj = json.loads(stripped)
        if isinstance(obj, dict):
            return obj, False, None
        return None, False, f"Top-level JSON is {type(obj).__name__}, expected dict"
    except json.JSONDecodeError:
        pass

    m_fence = _FENCE_RE.search(stripped)
    if m_fence:
        try:
            obj = json.loads(m_fence.group(1))
            if isinstance(obj, dict):
                return obj, True, None
        except json.JSONDecodeError:
            pass

    m_brace = _BRACE_RE.search(stripped)
    if m_brace:
        try:
            obj = json.loads(m_brace.group(1))
            if isinstance(obj, dict):
                return obj, True, None
        except json.JSONDecodeError as exc:
            return None, True, f"JSONDecodeError: {exc}"

    return None, False, "No valid JSON object found in output"


def parse_task_c_output(
    raw_output: str,
    allow_qr_code: bool = False,
) -> tuple[list[str], ParseStatus, list[str]]:
    """Strictly parse and validate a Task C model completion (`{"labels": [...]}`).

    Never fuzzy-matches or silently invents labels; unknown labels are recorded as errors and rejected.
    """
    obj, used_fence, err = _extract_json_object(raw_output)
    if obj is None:
        return [], "INVALID_JSON", [err or "JSON parse error"]

    if "labels" not in obj or not isinstance(obj["labels"], list):
        return [], "INVALID_SCHEMA", ["Missing 'labels' key or 'labels' is not a list"]

    allowed = set(get_allowed_task_c_labels(include_qr_code=allow_qr_code))
    valid_labels: list[str] = []
    errors: list[str] = []

    for item in obj["labels"]:
        if not isinstance(item, str):
            errors.append(f"Non-string label item: {item!r}")
            continue
        if item not in allowed:
            errors.append(f"Unknown or disallowed label: {item!r}")
            continue
        valid_labels.append(item)

    sorted_valid = sort_task_c_labels(valid_labels)
    if errors:
        has_unknown = any("Unknown or disallowed" in e for e in errors)
        status: ParseStatus = "UNKNOWN_LABEL" if has_unknown else "INVALID_SCHEMA"
        return sorted_valid, status, errors

    return sorted_valid, ("VALID_FENCED" if used_fence else "VALID_EXACT"), []


def parse_task_d_output(
    raw_output: str,
    allow_qr_code: bool = False,
) -> tuple[list[TaskDTargetItem], ParseStatus, list[str]]:
    """Strictly parse and validate a Task D model completion (`{"targets": [{"target": ..., "category": ..., "polarity": ...}]}`).

    Rejects any item that contains offset fields or unknown categories/polarities.
    """
    obj, used_fence, err = _extract_json_object(raw_output)
    if obj is None:
        return [], "INVALID_JSON", [err or "JSON parse error"]

    if "targets" not in obj or not isinstance(obj["targets"], list):
        return [], "INVALID_SCHEMA", ["Missing 'targets' key or 'targets' is not a list"]

    allowed_cats = set(MAIN_CATEGORIES_19) | ({"QR-Code"} if allow_qr_code else set())
    allowed_pols = set(POLARITIES)
    forbidden_offset_keys = {"from", "to", "start", "end", "offset", "token_index", "start_char", "end_char"}

    valid_targets: list[TaskDTargetItem] = []
    errors: list[str] = []

    for idx, item in enumerate(obj["targets"]):
        if not isinstance(item, dict):
            errors.append(f"Target item {idx} is not a dict")
            continue
        leaked_keys = forbidden_offset_keys.intersection(item.keys())
        if leaked_keys:
            errors.append(f"Target item {idx} contains forbidden offset keys: {sorted(leaked_keys)}")
            continue
        target_str = item.get("target")
        cat = item.get("category")
        pol = item.get("polarity")
        if not isinstance(target_str, str) or not target_str.strip() or target_str == "NULL":
            errors.append(f"Target item {idx} has invalid target string: {target_str!r}")
            continue
        if cat not in allowed_cats:
            errors.append(f"Target item {idx} has unknown category: {cat!r}")
            continue
        if pol not in allowed_pols:
            errors.append(f"Target item {idx} has unknown polarity: {pol!r}")
            continue
        valid_targets.append(TaskDTargetItem(target=target_str, category=cat, polarity=pol))

    if errors:
        has_unknown = any("unknown category" in e or "unknown polarity" in e for e in errors)
        status: ParseStatus = "UNKNOWN_LABEL" if has_unknown else "INVALID_SCHEMA"
        return valid_targets, status, errors

    return valid_targets, ("VALID_FENCED" if used_fence else "VALID_EXACT"), []
