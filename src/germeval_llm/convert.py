"""Derive Task C and Task D instruction datasets and stratified learning-curve subsets from canonical documents."""

from __future__ import annotations

import json
from pathlib import Path
import random
from typing import Iterable, Literal

from germeval_llm.prompts import (
    format_task_c_completion_json,
    format_task_d_completion_json,
    load_prompts_config,
    render_task_c_user_prompt,
    render_task_d_user_prompt,
)
from germeval_llm.schemas import (
    CanonicalDocument,
    CanonicalOpinion,
    ChatMessage,
    TaskCExample,
    TaskDExample,
    TaskDTargetItem,
)
from germeval_llm.taxonomy import (
    KNOWN_CATEGORY_MAIN_TYPOS,
    KNOWN_POLARITY_TYPOS,
    MAIN_CATEGORIES_19,
    sort_task_c_labels,
)


def normalize_opinion_for_derivation(
    op: CanonicalOpinion,
    apply_typo_patches: bool = True,
) -> tuple[str, str]:
    """Return (main_category, polarity) for an opinion, optionally applying documented training typo patches."""
    cat = op.category
    pol = op.polarity
    if apply_typo_patches:
        cat = KNOWN_CATEGORY_MAIN_TYPOS.get(cat, cat)
        pol = KNOWN_POLARITY_TYPOS.get(pol, pol)
    return cat, pol


def extract_task_c_label_sets(
    doc: CanonicalDocument,
    apply_typo_patches: bool = False,
) -> tuple[list[str], list[str]]:
    """Derive sorted (gold_labels ['Category:polarity'], gold_categories ['Category']) from a CanonicalDocument.

    Includes all opinions (both explicit and implicit NULL targets), matching GermevalEvaluator.java.
    Documents with zero opinions (including relevance=false) produce empty lists ([], []).
    """
    raw_pairs: set[str] = set()
    raw_cats: set[str] = set()
    for op in doc.opinions:
        cat, pol = normalize_opinion_for_derivation(op, apply_typo_patches=apply_typo_patches)
        raw_pairs.add(f"{cat}:{pol}")
        raw_cats.add(cat)

    sorted_pairs = sort_task_c_labels(raw_pairs)
    cat_rank = {c: i for i, c in enumerate(MAIN_CATEGORIES_19)}
    sorted_cats = sorted(raw_cats, key=lambda c: (cat_rank.get(c, 999), c))
    return sorted_pairs, sorted_cats


def derive_task_c_example(
    doc: CanonicalDocument,
    apply_typo_patches: bool = False,
    prompt_key: str = "p1_guidelines",
    prompts_cfg: dict | None = None,
) -> TaskCExample:
    """Convert a single CanonicalDocument into a TaskCExample."""
    gold_labels, gold_categories = extract_task_c_label_sets(doc, apply_typo_patches=apply_typo_patches)
    user_text = render_task_c_user_prompt(
        doc.text,
        prompt_key=prompt_key,
        few_shot_examples=None,
        prompts_cfg=prompts_cfg,
    )
    completion_str = format_task_c_completion_json(gold_labels)
    return TaskCExample(
        example_id=f"{doc.source_split}:{doc.source_ordinal:06d}",
        doc_id=doc.doc_id,
        source_split=doc.source_split,
        source_ordinal=doc.source_ordinal,
        text=doc.text,
        relevance=doc.relevance,
        gold_labels=gold_labels,
        gold_categories=gold_categories,
        prompt=[ChatMessage(role="user", content=user_text)],
        completion=[ChatMessage(role="assistant", content=completion_str)],
    )


def extract_task_d_targets(
    doc: CanonicalDocument,
    apply_typo_patches: bool = False,
    view: Literal["exact_and_star", "contiguous_only"] = "exact_and_star",
) -> list[TaskDTargetItem]:
    """Extract explicit target expressions (strictly NO offsets) in document appearance order.

    Deduplicates exact identical (target, category, polarity) triples on the same span while
    preserving repeated targets or multiple labels for the same target.
    """
    items: list[TaskDTargetItem] = []
    seen_span_triples: set[tuple[int, int, str, str, str]] = set()

    for op in doc.opinions:
        if op.is_implicit or op.target_xml == "NULL" or op.end_xml == 0:
            continue
        is_exact_sub = doc.text[op.start_xml : op.end_xml] == op.target_xml
        if view == "contiguous_only" and not is_exact_sub:
            continue
        if view == "exact_and_star" and not (is_exact_sub or op.is_discontinuous_star):
            continue

        cat, pol = normalize_opinion_for_derivation(op, apply_typo_patches=apply_typo_patches)
        span_key = (op.start_xml, op.end_xml, op.target_xml, cat, pol)
        if span_key in seen_span_triples:
            continue
        seen_span_triples.add(span_key)
        items.append(TaskDTargetItem(target=op.target_xml, category=cat, polarity=pol))

    return items


def derive_task_d_example(
    doc: CanonicalDocument,
    apply_typo_patches: bool = False,
    view: Literal["exact_and_star", "contiguous_only"] = "exact_and_star",
    prompt_key: str = "d1_guidelines",
    prompts_cfg: dict | None = None,
) -> TaskDExample:
    """Convert a single CanonicalDocument into a TaskDExample with target strings only (never offsets)."""
    gold_targets = extract_task_d_targets(doc, apply_typo_patches=apply_typo_patches, view=view)
    user_text = render_task_d_user_prompt(
        doc.text,
        prompt_key=prompt_key,
        few_shot_examples=None,
        prompts_cfg=prompts_cfg,
    )
    completion_str = format_task_d_completion_json(gold_targets)
    return TaskDExample(
        example_id=f"{doc.source_split}:{doc.source_ordinal:06d}",
        doc_id=doc.doc_id,
        source_split=doc.source_split,
        source_ordinal=doc.source_ordinal,
        text=doc.text,
        relevance=doc.relevance,
        view=view,
        gold_targets=gold_targets,
        prompt=[ChatMessage(role="user", content=user_text)],
        completion=[ChatMessage(role="assistant", content=completion_str)],
    )


def save_derived_jsonl(examples: Iterable[TaskCExample | TaskDExample], out_path: str | Path) -> Path:
    """Write derived Task C or Task D examples to JSONL."""
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex.model_dump(), ensure_ascii=False) + "\n")
    return out


def ensure_derived_datasets_ready(dataset_cfg_path: str | Path = "configs/dataset.yaml") -> None:
    """Ensure all canonical `.jsonl` and derived `data/derived/task_{c,d}/*.jsonl` files exist.

    On a fresh Git clone (e.g. in Google Colab), `data/canonical/*.jsonl.gz` (5 MB total) are
    tracked in Git while the 212 MB `data/derived/` files are ignored. This helper transparently
    decompresses `data/canonical/*.jsonl.gz` and materializes `data/derived/task_{c,d}/*.jsonl`
    in ~1.5 seconds without requiring the original raw XML files.
    """
    import yaml
    from germeval_llm.xml_parser import load_canonical_jsonl

    with Path(dataset_cfg_path).open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    prompts_cfg = load_prompts_config()
    for _split_name, scfg in cfg["splits"].items():
        can_path = Path(scfg["canonical_path"])
        c_path = Path(scfg["task_c_path"])
        d_path = Path(scfg["task_d_path"])
        if can_path.exists() and c_path.exists() and d_path.exists():
            continue
        docs = load_canonical_jsonl(can_path)
        apply_patches = bool(scfg.get("apply_typo_patches", False))
        if not c_path.exists():
            c_exs = [
                derive_task_c_example(d, apply_typo_patches=apply_patches, prompts_cfg=prompts_cfg)
                for d in docs
            ]
            save_derived_jsonl(c_exs, c_path)
        if not d_path.exists():
            d_exs = [
                derive_task_d_example(d, apply_typo_patches=apply_patches, prompts_cfg=prompts_cfg)
                for d in docs
            ]
            save_derived_jsonl(d_exs, d_path)



def select_few_shot_examples_task_c(
    train_examples: list[TaskCExample],
    n_shots: int,
    strategy: Literal["uniform", "stratified_diverse"] = "stratified_diverse",
    seed: int = 42,
) -> list[TaskCExample]:
    """Select deterministic few-shot demonstrations from training examples only."""
    if n_shots <= 0:
        return []
    rng = random.Random(seed)
    if strategy == "uniform":
        return rng.sample(train_examples, min(n_shots, len(train_examples)))

    # Stratified diverse: include 1 empty/irrelevant example, 1 single-label, and multi-label covering distinct categories
    empty_exs = [ex for ex in train_examples if len(ex.gold_labels) == 0 and 40 <= len(ex.text) <= 180]
    single_exs = [ex for ex in train_examples if len(ex.gold_labels) == 1 and 40 <= len(ex.text) <= 180]
    multi_exs = [ex for ex in train_examples if len(ex.gold_labels) >= 2 and 50 <= len(ex.text) <= 220]

    selected: list[TaskCExample] = []
    used_ids: set[str] = set()
    covered_cats: set[str] = set()

    if empty_exs:
        pick = empty_exs[rng.randrange(len(empty_exs))]
        selected.append(pick)
        used_ids.add(pick.example_id)

    if len(selected) < n_shots and single_exs:
        pick = single_exs[rng.randrange(len(single_exs))]
        selected.append(pick)
        used_ids.add(pick.example_id)
        covered_cats.update(pick.gold_categories)

    # Greedily add multi-label examples that maximize new category coverage
    candidates = list(multi_exs) + list(single_exs)
    rng.shuffle(candidates)
    while len(selected) < n_shots and candidates:
        best_ex = max(
            (c for c in candidates if c.example_id not in used_ids),
            key=lambda c: len(set(c.gold_categories) - covered_cats),
            default=None,
        )
        if best_ex is None:
            break
        selected.append(best_ex)
        used_ids.add(best_ex.example_id)
        covered_cats.update(best_ex.gold_categories)

    return selected[:n_shots]


def sample_learning_curve_subset(
    train_examples: list[TaskCExample],
    subset_size: int,
    strategy: Literal["uniform", "stratified"] = "stratified",
    seed: int = 42,
) -> list[TaskCExample]:
    """Sample a training subset of size N using either uniform random sampling or rare-first multilabel coverage."""
    if subset_size >= len(train_examples):
        return list(train_examples)
    rng = random.Random(seed)
    if strategy == "uniform":
        indices = sorted(rng.sample(range(len(train_examples)), subset_size))
        return [train_examples[i] for i in indices]

    # Multilabel rare-category-aware coverage + proportional empty-doc representation
    empty_pool = [ex for ex in train_examples if not ex.gold_labels]
    labeled_pool = [ex for ex in train_examples if ex.gold_labels]
    rng.shuffle(empty_pool)
    rng.shuffle(labeled_pool)

    target_empty = int(round(subset_size * (len(empty_pool) / len(train_examples))))
    selected: list[TaskCExample] = empty_pool[:target_empty]
    used_ids: set[str] = {ex.example_id for ex in selected}

    # Ensure every rare label present in train gets at least min_per_label examples before filling proportionally
    label_to_exs: dict[str, list[TaskCExample]] = {}
    for ex in labeled_pool:
        for lbl in ex.gold_labels:
            label_to_exs.setdefault(lbl, []).append(ex)

    rare_ordered_labels = sorted(label_to_exs.keys(), key=lambda k: len(label_to_exs[k]))
    min_per_label = max(1, subset_size // 100)
    label_counts: dict[str, int] = {k: 0 for k in rare_ordered_labels}

    for lbl in rare_ordered_labels:
        for ex in label_to_exs[lbl]:
            if len(selected) >= subset_size:
                break
            if label_counts[lbl] >= min_per_label:
                break
            if ex.example_id not in used_ids:
                selected.append(ex)
                used_ids.add(ex.example_id)
                for l2 in ex.gold_labels:
                    label_counts[l2] = label_counts.get(l2, 0) + 1

    for ex in labeled_pool:
        if len(selected) >= subset_size:
            break
        if ex.example_id not in used_ids:
            selected.append(ex)
            used_ids.add(ex.example_id)

    selected.sort(key=lambda x: x.source_ordinal)
    return selected
