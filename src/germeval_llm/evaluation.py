"""Authoritative Python evaluation metrics for GermEval 2017 Task C and Task D."""

from __future__ import annotations

from collections import Counter
from typing import Any, Sequence
import numpy as np

from germeval_llm.schemas import (
    AlignedTargetSpan,
    CanonicalDocument,
    PredictionRecord,
    TaskDTargetItem,
)
from germeval_llm.taxonomy import MAIN_CATEGORIES_19, get_allowed_task_c_labels


def compute_prf1(tp: int, fp: int, fn: int) -> dict[str, float]:
    """Compute precision, recall, and F1 from TP/FP/FN counts."""
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return {
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def evaluate_label_sets(
    gold_sets: Sequence[set[str]],
    pred_sets: Sequence[set[str]],
    label_inventory: Sequence[str] | None = None,
    ignore_empty_gold_docs: bool = False,
) -> dict[str, Any]:
    """Evaluate document-level set predictions (used for both Task C1 and Task C2).

    Args:
        gold_sets: Per-document set of gold labels.
        pred_sets: Per-document set of predicted labels.
        label_inventory: Fixed inventory for per-label and macro metrics.
        ignore_empty_gold_docs: When True, replicates the legacy Java `GermevalEvaluator.java:140`
            quirk where documents with 0 gold opinions are skipped entirely (so false positives on
            empty gold documents are not counted). When False (default), computes strict micro/macro-F1.
    """
    if len(gold_sets) != len(pred_sets):
        raise ValueError(f"Length mismatch: gold={len(gold_sets)} vs pred={len(pred_sets)}")

    total_tp = 0
    total_fp = 0
    total_fn = 0
    exact_matches = 0
    evaluated_docs = 0

    per_label_counts: dict[str, list[int]] = {}
    if label_inventory:
        for lbl in label_inventory:
            per_label_counts[lbl] = [0, 0, 0]  # tp, fp, fn

    for g_set, p_set in zip(gold_sets, pred_sets):
        if g_set == p_set:
            exact_matches += 1
        if ignore_empty_gold_docs and len(g_set) == 0:
            continue
        evaluated_docs += 1

        tp_items = g_set.intersection(p_set)
        fp_items = p_set - g_set
        fn_items = g_set - p_set

        total_tp += len(tp_items)
        total_fp += len(fp_items)
        total_fn += len(fn_items)

        for lbl in tp_items:
            per_label_counts.setdefault(lbl, [0, 0, 0])[0] += 1
        for lbl in fp_items:
            per_label_counts.setdefault(lbl, [0, 0, 0])[1] += 1
        for lbl in fn_items:
            per_label_counts.setdefault(lbl, [0, 0, 0])[2] += 1

    micro = compute_prf1(total_tp, total_fp, total_fn)

    per_label_metrics: dict[str, dict[str, float]] = {}
    active_f1s: list[float] = []
    all_f1s: list[float] = []

    for lbl, (ltp, lfp, lfn) in sorted(per_label_counts.items()):
        m = compute_prf1(ltp, lfp, lfn)
        m["support"] = int(ltp + lfn)
        per_label_metrics[lbl] = m
        all_f1s.append(m["f1"])
        if (ltp + lfn) > 0:
            active_f1s.append(m["f1"])

    macro_f1_active = float(np.mean(active_f1s)) if active_f1s else 0.0
    macro_f1_all = float(np.mean(all_f1s)) if all_f1s else 0.0
    exact_match_acc = exact_matches / len(gold_sets) if gold_sets else 0.0

    return {
        "num_documents": len(gold_sets),
        "num_evaluated_documents": evaluated_docs,
        "exact_set_accuracy": float(exact_match_acc),
        "micro_precision": micro["precision"],
        "micro_recall": micro["recall"],
        "micro_f1": micro["f1"],
        "macro_f1_active": macro_f1_active,
        "macro_f1_all_inventory": macro_f1_all,
        "tp": micro["tp"],
        "fp": micro["fp"],
        "fn": micro["fn"],
        "per_label": per_label_metrics,
    }


def bootstrap_micro_f1_ci(
    gold_sets: Sequence[set[str]],
    pred_sets: Sequence[set[str]],
    n_bootstrap: int = 1000,
    seed: int = 42,
    ignore_empty_gold_docs: bool = False,
) -> dict[str, float]:
    """Compute paired bootstrap 95% confidence interval for document-level Micro-F1."""
    n = len(gold_sets)
    if n == 0:
        return {"mean": 0.0, "ci95_low": 0.0, "ci95_high": 0.0, "std": 0.0}

    tp_arr = np.zeros(n, dtype=np.int64)
    fp_arr = np.zeros(n, dtype=np.int64)
    fn_arr = np.zeros(n, dtype=np.int64)

    for i, (g, p) in enumerate(zip(gold_sets, pred_sets)):
        if ignore_empty_gold_docs and len(g) == 0:
            continue
        tp_arr[i] = len(g.intersection(p))
        fp_arr[i] = len(p - g)
        fn_arr[i] = len(g - p)

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_bootstrap, n))
    tp_sum = tp_arr[idx].sum(axis=1)
    fp_sum = fp_arr[idx].sum(axis=1)
    fn_sum = fn_arr[idx].sum(axis=1)

    denom = (2 * tp_sum + fp_sum + fn_sum).astype(np.float64)
    scores = np.divide(2.0 * tp_sum, denom, out=np.zeros_like(denom), where=denom > 0)

    return {
        "mean": float(np.mean(scores)),
        "std": float(np.std(scores)),
        "ci95_low": float(np.percentile(scores, 2.5)),
        "ci95_high": float(np.percentile(scores, 97.5)),
    }


def paired_bootstrap_delta_f1_ci(
    gold_sets: Sequence[set[str]],
    baseline_pred_sets: Sequence[set[str]],
    ft_pred_sets: Sequence[set[str]],
    n_bootstrap: int = 1000,
    seed: int = 42,
    ignore_empty_gold_docs: bool = False,
) -> dict[str, float]:
    """Compute paired document-level bootstrap 95% confidence interval for ΔF1 = F1_ft - F1_baseline."""
    n = len(gold_sets)
    if len(baseline_pred_sets) != n or len(ft_pred_sets) != n:
        raise ValueError(
            f"Length mismatch in paired bootstrap: gold={n}, "
            f"baseline={len(baseline_pred_sets)}, ft={len(ft_pred_sets)}"
        )
    if n == 0:
        return {
            "point_baseline_f1": 0.0,
            "point_ft_f1": 0.0,
            "point_delta_f1": 0.0,
            "delta_mean": 0.0,
            "delta_std": 0.0,
            "delta_ci95_low": 0.0,
            "delta_ci95_high": 0.0,
            "baseline_ci95_low": 0.0,
            "baseline_ci95_high": 0.0,
            "ft_ci95_low": 0.0,
            "ft_ci95_high": 0.0,
            "prob_improvement": 0.0,
        }

    b_tp = np.zeros(n, dtype=np.int64)
    b_fp = np.zeros(n, dtype=np.int64)
    b_fn = np.zeros(n, dtype=np.int64)
    q_tp = np.zeros(n, dtype=np.int64)
    q_fp = np.zeros(n, dtype=np.int64)
    q_fn = np.zeros(n, dtype=np.int64)

    for i, (g, b, q) in enumerate(zip(gold_sets, baseline_pred_sets, ft_pred_sets)):
        if ignore_empty_gold_docs and len(g) == 0:
            continue
        b_tp[i] = len(g.intersection(b))
        b_fp[i] = len(b - g)
        b_fn[i] = len(g - b)
        q_tp[i] = len(g.intersection(q))
        q_fp[i] = len(q - g)
        q_fn[i] = len(g - q)

    point_b = compute_prf1(int(b_tp.sum()), int(b_fp.sum()), int(b_fn.sum()))["f1"]
    point_q = compute_prf1(int(q_tp.sum()), int(q_fp.sum()), int(q_fn.sum()))["f1"]

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_bootstrap, n))

    b_tp_s = b_tp[idx].sum(axis=1)
    b_fp_s = b_fp[idx].sum(axis=1)
    b_fn_s = b_fn[idx].sum(axis=1)
    b_denom = (2 * b_tp_s + b_fp_s + b_fn_s).astype(np.float64)
    b_scores = np.divide(2.0 * b_tp_s, b_denom, out=np.zeros_like(b_denom), where=b_denom > 0)

    q_tp_s = q_tp[idx].sum(axis=1)
    q_fp_s = q_fp[idx].sum(axis=1)
    q_fn_s = q_fn[idx].sum(axis=1)
    q_denom = (2 * q_tp_s + q_fp_s + q_fn_s).astype(np.float64)
    q_scores = np.divide(2.0 * q_tp_s, q_denom, out=np.zeros_like(q_denom), where=q_denom > 0)

    deltas = q_scores - b_scores

    return {
        "point_baseline_f1": float(point_b),
        "point_ft_f1": float(point_q),
        "point_delta_f1": float(point_q - point_b),
        "delta_mean": float(np.mean(deltas)),
        "delta_std": float(np.std(deltas)),
        "delta_ci95_low": float(np.percentile(deltas, 2.5)),
        "delta_ci95_high": float(np.percentile(deltas, 97.5)),
        "baseline_ci95_low": float(np.percentile(b_scores, 2.5)),
        "baseline_ci95_high": float(np.percentile(b_scores, 97.5)),
        "ft_ci95_low": float(np.percentile(q_scores, 2.5)),
        "ft_ci95_high": float(np.percentile(q_scores, 97.5)),
        "prob_improvement": float(np.mean(deltas > 0.0)),
    }


def align_predictions(
    docs: Sequence[CanonicalDocument],
    predictions: Sequence[PredictionRecord],
    expected_split: str | None = None,
    allow_subset: bool = False,
) -> list[PredictionRecord]:
    """Verify and align predictions to canonical documents by doc_id.

    Rejects duplicate prediction document IDs, missing document IDs, and (unless allow_subset=True)
    unexpected extra document IDs. Also verifies split metadata when expected_split is provided.
    """
    expected_ids = [d.doc_id for d in docs]
    if len(expected_ids) != len(set(expected_ids)):
        raise ValueError("Duplicate document IDs in evaluation documents")

    prediction_ids = [p.doc_id for p in predictions]
    if len(prediction_ids) != len(set(prediction_ids)):
        raise ValueError("Duplicate prediction document IDs")

    by_id = {p.doc_id: p for p in predictions}
    if allow_subset:
        missing = set(expected_ids) - set(by_id)
        if missing:
            raise ValueError(f"Prediction mismatch: {len(missing)} missing, 0 unexpected")
    else:
        if set(expected_ids) != set(by_id):
            missing = set(expected_ids) - set(by_id)
            extra = set(by_id) - set(expected_ids)
            raise ValueError(
                f"Prediction mismatch: {len(missing)} missing, "
                f"{len(extra)} unexpected"
            )

    aligned = [by_id[doc_id] for doc_id in expected_ids]
    if expected_split is not None:
        mismatched_splits = {p.split for p in aligned if p.split != expected_split}
        if mismatched_splits:
            raise ValueError(
                f"Prediction split mismatch: expected {expected_split!r}, found {sorted(mismatched_splits)}"
            )
    return aligned


def verify_experiment_manifest(
    manifest: dict[str, Any],
    expected_split: str,
    expected_prompt_id: str,
    expected_eval_docs: Sequence[CanonicalDocument] | None = None,
    expected_base_model: str | None = None,
) -> dict[str, Any]:
    """Verify that a replay or live experiment manifest matches the active evaluation configuration."""
    required_keys = {
        "experiment_id",
        "base_model",
        "prompt_id",
        "training_split",
        "evaluation_split",
        "seed",
    }
    missing_keys = required_keys - set(manifest.keys())
    if missing_keys:
        raise ValueError(f"Replay manifest mismatch: missing required keys {sorted(missing_keys)}")

    if manifest["evaluation_split"] != expected_split:
        raise ValueError(
            f"Replay manifest mismatch: evaluation_split={manifest['evaluation_split']!r} "
            f"does not match expected {expected_split!r}"
        )

    if str(manifest["prompt_id"]).lower() != expected_prompt_id.lower():
        raise ValueError(
            f"Replay manifest mismatch: prompt_id={manifest['prompt_id']!r} "
            f"does not match expected {expected_prompt_id!r}"
        )

    if expected_base_model is not None and manifest["base_model"] != expected_base_model:
        raise ValueError(
            f"Replay manifest mismatch: base_model={manifest['base_model']!r} "
            f"does not match expected {expected_base_model!r}"
        )

    if expected_eval_docs is not None:
        expected_ids = [d.doc_id for d in expected_eval_docs]
        if "evaluation_num_docs" in manifest and int(manifest["evaluation_num_docs"]) != len(expected_ids):
            raise ValueError(
                f"Replay manifest mismatch: evaluation_num_docs={manifest['evaluation_num_docs']} "
                f"does not match len(eval_docs)={len(expected_ids)}"
            )
        if "evaluation_doc_ids" in manifest and list(manifest["evaluation_doc_ids"]) != expected_ids:
            raise ValueError(
                "Replay manifest mismatch: evaluation_doc_ids do not match expected evaluation documents"
            )

    return manifest


def compute_java_fscore_micro_f1(entries: Sequence[tuple[str, str]]) -> dict[str, Any]:
    """Exact port of `de.germeval2017.Evaluation.EvaluationUtil.Fscore.java`.

    In `Fscore.java:37-69`:
    `categories = getDistinctLabels(data)` includes dummy placeholder labels like `"none"`,
    `"none_sentiment"`, and `"missmatch"` registered via `register(gold, "none")` (for FN)
    and `register("none", pred)` (for FP).
    Because `"none"` never appears as `gold == pred`, its per-label `tp = 0`, `fp = total_FNs`,
    and `fn = total_FPs`. Summing `tp_sum`, `fp_sum`, and `fn_sum` over all labels in `categories`
    including `"none"` therefore doubles both `fp_sum` and `fn_sum`:
        `micro_fscore_java = 2 * TP / (2 * TP + 2 * (FP + FN)) = TP / (TP + FP + FN)` (Jaccard index!)
    whenever at least one FP or FN is present!
    """
    distinct_labels: set[str] = set()
    for g, p in entries:
        distinct_labels.add(g)
        distinct_labels.add(p)

    tp_sum = 0
    fp_sum = 0
    fn_sum = 0
    for cat in distinct_labels:
        for g, p in entries:
            if g == cat:
                if g == p:
                    tp_sum += 1
                else:
                    fn_sum += 1
            else:
                if p == cat:
                    fp_sum += 1

    micro_p = tp_sum / (tp_sum + fp_sum) if (tp_sum + fp_sum) > 0 else 0.0
    micro_r = tp_sum / (tp_sum + fn_sum) if (tp_sum + fn_sum) > 0 else 0.0
    micro_f1 = (2.0 * micro_p * micro_r / (micro_p + micro_r)) if (micro_p + micro_r) > 0 else 0.0
    return {
        "num_entries": len(entries),
        "tp_sum": tp_sum,
        "fp_sum": fp_sum,
        "fn_sum": fn_sum,
        "micro_f1": float(micro_f1),
    }


def compute_java_category_evaluator(
    gold_sets: Sequence[set[str]],
    pred_sets: Sequence[set[str]],
    none_token: str = "none",
) -> dict[str, Any]:
    """Exact port of `GermevalEvaluator.java:evaluateCategory` + `Fscore.java` for TSV evaluation."""
    entries: list[tuple[str, str]] = []
    for g_set, p_set in zip(gold_sets, pred_sets):
        for g in g_set:
            if g in p_set:
                entries.append((g, g))
            else:
                entries.append((g, none_token))
        for p in p_set:
            if p not in g_set:
                entries.append((none_token, p))
    return compute_java_fscore_micro_f1(entries)


def evaluate_task_c_predictions(
    canonical_docs: Sequence[CanonicalDocument],
    predictions: Sequence[PredictionRecord],
    apply_gold_typo_patches: bool = False,
    n_bootstrap: int = 500,
) -> dict[str, Any]:
    """Full Task C evaluation reporting both C1 (Category) and C2 (Category+Sentiment)
    under both Strict Python scoring and Legacy Java-compatible scoring (`EvaluationScript.jar`).
    """
    pred_by_id = {p.doc_id: p for p in predictions}

    gold_c1_raw: list[set[str]] = []
    gold_c2_raw: list[set[str]] = []
    pred_c1: list[set[str]] = []
    pred_c2: list[set[str]] = []

    status_counts: Counter[str] = Counter()
    empty_pred_count = 0
    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_latency_ms = 0.0

    for doc in canonical_docs:
        if apply_gold_typo_patches:
            from germeval_llm.convert import extract_task_c_label_sets

            g_lbls, g_cats = extract_task_c_label_sets(doc, apply_typo_patches=True)
            gold_c1_raw.append(set(g_cats))
            gold_c2_raw.append(set(g_lbls))
        else:
            # Exact raw XML attributes as seen by GermevalEvaluator.java
            gold_c1_raw.append({op.category for op in doc.opinions})
            gold_c2_raw.append({f"{op.category}:{op.polarity}" for op in doc.opinions})

        pred = pred_by_id.get(doc.doc_id)
        if pred is None:
            pred_c1.append(set())
            pred_c2.append(set())
            status_counts["MISSING_PREDICTION"] += 1
        else:
            status_counts[pred.parse_status] += 1
            p_set = set(pred.parsed_labels)
            p_cats = {lbl.split(":", 1)[0] for lbl in p_set}
            pred_c2.append(p_set)
            pred_c1.append(p_cats)
            if len(p_set) == 0:
                empty_pred_count += 1
            total_prompt_tokens += pred.prompt_tokens
            total_completion_tokens += pred.completion_tokens
            total_latency_ms += pred.latency_ms

    c1_strict = evaluate_label_sets(
        gold_c1_raw, pred_c1, label_inventory=MAIN_CATEGORIES_19, ignore_empty_gold_docs=False
    )
    c2_strict = evaluate_label_sets(
        gold_c2_raw,
        pred_c2,
        label_inventory=get_allowed_task_c_labels(include_qr_code=False),
        ignore_empty_gold_docs=False,
    )
    c1_java = compute_java_category_evaluator(gold_c1_raw, pred_c1, none_token="none")
    c2_java = compute_java_category_evaluator(gold_c2_raw, pred_c2, none_token="none_sentiment")

    c2_ci = bootstrap_micro_f1_ci(gold_c2_raw, pred_c2, n_bootstrap=n_bootstrap, ignore_empty_gold_docs=False)

    n_docs = len(canonical_docs)
    valid_count = status_counts["VALID_EXACT"] + status_counts["VALID_FENCED"]

    return {
        "num_documents": n_docs,
        "schema_validity_rate": valid_count / n_docs if n_docs else 0.0,
        "valid_exact_rate": status_counts["VALID_EXACT"] / n_docs if n_docs else 0.0,
        "empty_prediction_rate": empty_pred_count / n_docs if n_docs else 0.0,
        "parse_status_counts": dict(status_counts),
        "task_c1_category_strict": c1_strict,
        "task_c2_category_sentiment_strict": c2_strict,
        "task_c1_category_java_compat": c1_java,
        "task_c2_category_sentiment_java_compat": c2_java,
        "task_c2_strict_bootstrap_ci95": c2_ci,
        "efficiency": {
            "total_prompt_tokens": total_prompt_tokens,
            "total_completion_tokens": total_completion_tokens,
            "mean_prompt_tokens_per_doc": total_prompt_tokens / n_docs if n_docs else 0.0,
            "mean_completion_tokens_per_doc": total_completion_tokens / n_docs if n_docs else 0.0,
            "prompt_tokens_per_1k_docs": (total_prompt_tokens * 1000.0 / n_docs) if n_docs else 0.0,
            "total_latency_ms": total_latency_ms,
            "mean_latency_ms_per_doc": total_latency_ms / n_docs if n_docs else 0.0,
        },
    }


def compare_task_c_models(
    canonical_docs: Sequence[CanonicalDocument],
    baseline_preds: Sequence[PredictionRecord],
    ft_preds: Sequence[PredictionRecord],
    expected_split: str | None = None,
    apply_gold_typo_patches: bool = False,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> dict[str, Any]:
    """Align baseline and fine-tuned predictions to canonical_docs and compute paired metrics + ΔF1 bootstrap CIs."""
    aligned_b = align_predictions(canonical_docs, baseline_preds, expected_split=expected_split)
    aligned_q = align_predictions(canonical_docs, ft_preds, expected_split=expected_split)

    b_metrics = evaluate_task_c_predictions(
        canonical_docs, aligned_b, apply_gold_typo_patches=apply_gold_typo_patches, n_bootstrap=n_bootstrap
    )
    q_metrics = evaluate_task_c_predictions(
        canonical_docs, aligned_q, apply_gold_typo_patches=apply_gold_typo_patches, n_bootstrap=n_bootstrap
    )

    gold_c1: list[set[str]] = []
    gold_c2: list[set[str]] = []
    for doc in canonical_docs:
        if apply_gold_typo_patches:
            from germeval_llm.convert import extract_task_c_label_sets

            g_lbls, g_cats = extract_task_c_label_sets(doc, apply_typo_patches=True)
            gold_c1.append(set(g_cats))
            gold_c2.append(set(g_lbls))
        else:
            gold_c1.append({op.category for op in doc.opinions})
            gold_c2.append({f"{op.category}:{op.polarity}" for op in doc.opinions})

    b_c2_sets = [set(p.parsed_labels) for p in aligned_b]
    b_c1_sets = [{lbl.split(":", 1)[0] for lbl in s} for s in b_c2_sets]
    q_c2_sets = [set(p.parsed_labels) for p in aligned_q]
    q_c1_sets = [{lbl.split(":", 1)[0] for lbl in s} for s in q_c2_sets]

    c1_paired_ci = paired_bootstrap_delta_f1_ci(
        gold_c1, b_c1_sets, q_c1_sets, n_bootstrap=n_bootstrap, seed=seed
    )
    c2_paired_ci = paired_bootstrap_delta_f1_ci(
        gold_c2, b_c2_sets, q_c2_sets, n_bootstrap=n_bootstrap, seed=seed
    )

    return {
        "num_documents": len(canonical_docs),
        "baseline": b_metrics,
        "finetuned": q_metrics,
        "deltas": {
            "valid_exact_rate": float(q_metrics["valid_exact_rate"] - b_metrics["valid_exact_rate"]),
            "schema_validity_rate": float(q_metrics["schema_validity_rate"] - b_metrics["schema_validity_rate"]),
            "c1_strict_micro_f1": float(
                q_metrics["task_c1_category_strict"]["micro_f1"]
                - b_metrics["task_c1_category_strict"]["micro_f1"]
            ),
            "c2_strict_micro_f1": float(
                q_metrics["task_c2_category_sentiment_strict"]["micro_f1"]
                - b_metrics["task_c2_category_sentiment_strict"]["micro_f1"]
            ),
            "c2_java_compat_micro_f1": float(
                q_metrics["task_c2_category_sentiment_java_compat"]["micro_f1"]
                - b_metrics["task_c2_category_sentiment_java_compat"]["micro_f1"]
            ),
            "exact_set_accuracy": float(
                q_metrics["task_c2_category_sentiment_strict"]["exact_set_accuracy"]
                - b_metrics["task_c2_category_sentiment_strict"]["exact_set_accuracy"]
            ),
        },
        "paired_bootstrap_c1": c1_paired_ci,
        "paired_bootstrap_c2": c2_paired_ci,
    }


def _java_get_max(end: int, doc_text: str) -> int:
    """Exact port of `OTEMatcher.java:getMax(int end, String docText)`."""
    upper_text = doc_text[end:].strip()
    if " " in upper_text:
        next_token = upper_text.split(" ")[0]
        return len(next_token) + 1
    return len(doc_text)


def _java_get_min(begin: int, doc_text: str) -> int:
    """Exact port of `OTEMatcher.java:getMin(int begin, String docText)`."""
    lower_text = doc_text[:begin]
    if " " in lower_text:
        tokens = lower_text.split(" ")
        preceding_token = tokens[-1]
        return len(preceding_token) + 1
    return 0


def evaluate_task_d_predictions(
    canonical_docs: Sequence[CanonicalDocument],
    predictions: Sequence[PredictionRecord],
) -> dict[str, Any]:
    """Evaluate Task D predictions on:
    1. String-level multiset F1: (target_string, category, polarity) and (target_string, category)
    2. Explicit-target Span F1: exact (start, end, category) and overlap (start, end, category),
       where unaligned predictions count as false positives.
    """
    pred_by_id = {p.doc_id: p for p in predictions}

    str_cat_pol_tp = str_cat_pol_fp = str_cat_pol_fn = 0
    str_cat_tp = str_cat_fp = str_cat_fn = 0
    span_exact_tp = span_exact_fp = span_exact_fn = 0
    span_overlap_tp = span_overlap_fp = span_overlap_fn = 0

    java_exact_entries: list[tuple[str, str]] = []
    java_overlap_entries: list[tuple[str, str]] = []
    alignment_counts: Counter[str] = Counter()

    for doc in canonical_docs:
        explicit_gold_ops = [
            op for op in doc.opinions if (not op.is_implicit and op.target_xml != "NULL" and op.end_xml > 0)
        ]

        pred_rec = pred_by_id.get(doc.doc_id)
        pred_targets: list[TaskDTargetItem] = pred_rec.parsed_targets if pred_rec else []
        aligned_spans: list[AlignedTargetSpan] = pred_rec.aligned_spans if pred_rec else []
        valid_aligned_spans = [
            sp for sp in aligned_spans if sp.start_char is not None and sp.end_char is not None
        ]

        for sp in aligned_spans:
            alignment_counts[sp.alignment_status] += 1

        # 1a. String-level multiset: (target, category, polarity)
        g_counter_cp = Counter((op.target_xml, op.category, op.polarity) for op in explicit_gold_ops)
        p_counter_cp = Counter((t.target, t.category, t.polarity) for t in pred_targets)
        tp_cp = sum((g_counter_cp & p_counter_cp).values())
        str_cat_pol_tp += tp_cp
        str_cat_pol_fp += sum(p_counter_cp.values()) - tp_cp
        str_cat_pol_fn += sum(g_counter_cp.values()) - tp_cp

        # 1b. String-level multiset: (target, category)
        g_counter_c = Counter((op.target_xml, op.category) for op in explicit_gold_ops)
        p_counter_c = Counter((t.target, t.category) for t in pred_targets)
        tp_c = sum((g_counter_c & p_counter_c).values())
        str_cat_tp += tp_c
        str_cat_fp += sum(p_counter_c.values()) - tp_c
        str_cat_fn += sum(g_counter_c.values()) - tp_c

        # 2a. Span-level exact matching (replicating OTEMatcher.registerExactOTEs on explicit targets)
        used_pred_exact: set[int] = set()
        for g_op in explicit_gold_ops:
            matched_idx: int | None = None
            for p_idx, p_sp in enumerate(aligned_spans):
                if p_idx in used_pred_exact or p_sp.start_char is None or p_sp.end_char is None:
                    continue
                if (
                    p_sp.start_char == g_op.start_xml
                    and p_sp.end_char == g_op.end_xml
                    and p_sp.category == g_op.category
                ):
                    matched_idx = p_idx
                    break
            if matched_idx is not None:
                span_exact_tp += 1
                used_pred_exact.add(matched_idx)
                if len(valid_aligned_spans) > 0:
                    java_exact_entries.append((g_op.category_full, g_op.category_full))
            else:
                span_exact_fn += 1
                if len(valid_aligned_spans) > 0:
                    java_exact_entries.append((g_op.category_full, "missmatch"))
                else:
                    java_exact_entries.append((g_op.category, "none"))
        # Every unmatched prediction (including unaligned ones in strict Python) is a false positive
        span_exact_fp += len(aligned_spans) - len(used_pred_exact)
        if len(explicit_gold_ops) > 0 and len(valid_aligned_spans) > 0:
            for p_idx, p_sp in enumerate(aligned_spans):
                if p_idx not in used_pred_exact and p_sp.start_char is not None and p_sp.end_char is not None:
                    java_exact_entries.append(("missmatch", p_sp.category))

        # 2b. Span-level overlap matching (replicating OTEMatcher.registerOverlaptOTEs on explicit targets)
        used_pred_overlap: set[int] = set()
        for g_op in explicit_gold_ops:
            lb_min = _java_get_min(g_op.start_xml, doc.text)
            lb_max = _java_get_max(g_op.start_xml, doc.text)
            ub_min = _java_get_min(g_op.end_xml, doc.text)
            ub_max = _java_get_max(g_op.end_xml, doc.text)

            matched_idx = None
            for p_idx, p_sp in enumerate(aligned_spans):
                if p_idx in used_pred_overlap or p_sp.start_char is None or p_sp.end_char is None:
                    continue
                if (
                    g_op.start_xml - lb_min <= p_sp.start_char <= g_op.start_xml + lb_max
                    and g_op.end_xml - ub_min <= p_sp.end_char <= g_op.end_xml + ub_max
                    and p_sp.category == g_op.category
                ):
                    matched_idx = p_idx
                    break
            if matched_idx is not None:
                span_overlap_tp += 1
                used_pred_overlap.add(matched_idx)
                if len(valid_aligned_spans) > 0:
                    java_overlap_entries.append((g_op.category, g_op.category))
            else:
                span_overlap_fn += 1
                if len(valid_aligned_spans) > 0:
                    java_overlap_entries.append((g_op.category, "none"))
                else:
                    java_overlap_entries.append((g_op.category, "missmatch"))
        span_overlap_fp += len(aligned_spans) - len(used_pred_overlap)
        if len(explicit_gold_ops) > 0 and len(valid_aligned_spans) > 0:
            for p_idx, p_sp in enumerate(aligned_spans):
                if p_idx not in used_pred_overlap and p_sp.start_char is not None and p_sp.end_char is not None:
                    java_overlap_entries.append(("none", p_sp.category))

    total_aligned_attempts = sum(alignment_counts.values())

    return {
        "num_documents": len(canonical_docs),
        "string_multiset_category_polarity": compute_prf1(str_cat_pol_tp, str_cat_pol_fp, str_cat_pol_fn),
        "string_multiset_category_only": compute_prf1(str_cat_tp, str_cat_fp, str_cat_fn),
        "span_explicit_exact": compute_prf1(span_exact_tp, span_exact_fp, span_exact_fn),
        "span_explicit_overlap": compute_prf1(span_overlap_tp, span_overlap_fp, span_overlap_fn),
        "span_explicit_exact_java_compat": compute_java_fscore_micro_f1(java_exact_entries),
        "span_explicit_overlap_java_compat": compute_java_fscore_micro_f1(java_overlap_entries),
        "alignment_breakdown": {
            "total_predicted_targets": total_aligned_attempts,
            "counts": dict(alignment_counts),
            "rates": {
                k: (v / total_aligned_attempts if total_aligned_attempts > 0 else 0.0)
                for k, v in alignment_counts.items()
            },
        },
    }
