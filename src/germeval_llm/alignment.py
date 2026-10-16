"""Deterministic post-processing alignment from Task D target strings to character offsets."""

from __future__ import annotations

from typing import Sequence

from germeval_llm.schemas import AlignedTargetSpan, TaskDTargetItem


def find_all_exact_occurrences(text: str, sub: str) -> list[tuple[int, int]]:
    """Return all exact (start, end) character occurrences of non-empty substring `sub` in `text`."""
    if not sub:
        return []
    spans: list[tuple[int, int]] = []
    start = 0
    while True:
        idx = text.find(sub, start)
        if idx == -1:
            break
        spans.append((idx, idx + len(sub)))
        start = idx + 1
    return spans


def align_discontinuous_star_target(text: str, target_star: str) -> tuple[int, int] | None:
    """Align a discontinuous 'part1 * part2' target string to its minimal outer bounding span (start_1, end_k)."""
    parts = [p.strip() for p in target_star.split("*") if p.strip()]
    if len(parts) < 2:
        return None

    first_occurrences = find_all_exact_occurrences(text, parts[0])
    if not first_occurrences:
        return None

    best_span: tuple[int, int] | None = None
    best_length = len(text) + 1

    for s1, e1 in first_occurrences:
        pos = e1
        matched_all = True
        last_end = e1
        for part in parts[1:]:
            idx = text.find(part, pos)
            if idx == -1:
                matched_all = False
                break
            last_end = idx + len(part)
            pos = last_end
        if matched_all:
            span_len = last_end - s1
            if span_len < best_length:
                best_length = span_len
                best_span = (s1, last_end)

    return best_span


def align_targets_to_text(
    text: str,
    predicted_targets: Sequence[TaskDTargetItem],
) -> list[AlignedTargetSpan]:
    """Deterministically align a list of predicted Task D target strings to character spans in `text`.

    Rules (Section 6.2):
    1. If '*' not in target:
       - Find all exact occurrences in `text`.
       - 0 occurrences -> `unaligned` (start_char=None, end_char=None).
       - 1 occurrence -> `unique_contiguous`.
       - >1 occurrences -> `ambiguous_contiguous`; deterministically assign the leftmost occurrence
         not yet assigned to the same (target, category) in this document (falling back to index 0 if all used).
    2. If '*' in target:
       - Find ordered non-overlapping occurrences of the '*'-separated parts and assign the outer
         bounding span `(start_first, end_last)` with status `discontinuous_star` (or `unaligned` if not found).
    3. Separate category/polarity labels for the same target string are preserved on their span.
    """
    aligned: list[AlignedTargetSpan] = []
    # Track how many times a given (target_str, category) has been assigned to advance across repeated occurrences
    used_occurrence_cursor: dict[tuple[str, str], int] = {}

    for item in predicted_targets:
        t_str = item.target
        if "*" in t_str:
            bound = align_discontinuous_star_target(text, t_str)
            if bound is not None:
                aligned.append(
                    AlignedTargetSpan(
                        target=t_str,
                        category=item.category,
                        polarity=item.polarity,
                        alignment_status="discontinuous_star",
                        start_char=bound[0],
                        end_char=bound[1],
                        occurrence_count=1,
                        assigned_occurrence_index=0,
                    )
                )
                continue

        occurrences = find_all_exact_occurrences(text, t_str)
        n_occ = len(occurrences)
        if n_occ == 0:
            aligned.append(
                AlignedTargetSpan(
                    target=t_str,
                    category=item.category,
                    polarity=item.polarity,
                    alignment_status="unaligned",
                    start_char=None,
                    end_char=None,
                    occurrence_count=0,
                    assigned_occurrence_index=None,
                )
            )
        elif n_occ == 1:
            s, e = occurrences[0]
            aligned.append(
                AlignedTargetSpan(
                    target=t_str,
                    category=item.category,
                    polarity=item.polarity,
                    alignment_status="unique_contiguous",
                    start_char=s,
                    end_char=e,
                    occurrence_count=1,
                    assigned_occurrence_index=0,
                )
            )
        else:
            key = (t_str, item.category)
            occ_idx = used_occurrence_cursor.get(key, 0)
            chosen_idx = min(occ_idx, n_occ - 1)
            used_occurrence_cursor[key] = occ_idx + 1
            s, e = occurrences[chosen_idx]
            aligned.append(
                AlignedTargetSpan(
                    target=t_str,
                    category=item.category,
                    polarity=item.polarity,
                    alignment_status="ambiguous_contiguous",
                    start_char=s,
                    end_char=e,
                    occurrence_count=n_occ,
                    assigned_occurrence_index=chosen_idx,
                )
            )

    return aligned
