"""Plotting, benchmark table consolidation, and error-analysis utilities for notebooks and reports."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from germeval_llm.schemas import CanonicalDocument, PredictionRecord


def setup_plot_style() -> None:
    """Configure consistent publication-quality Matplotlib/Seaborn style."""
    sns.set_theme(style="whitegrid", palette="colorblind")
    plt.rcParams["figure.dpi"] = 120
    plt.rcParams["savefig.dpi"] = 150
    plt.rcParams["font.size"] = 10


def plot_dataset_category_distribution(
    stats_summary: dict[str, Any],
    out_path: str | Path = "results/figures/dataset_category_distribution.png",
) -> Path:
    """Plot horizontal bar chart of main category frequencies in train_v1.4 and dev_v1.4."""
    setup_plot_style()
    train_cats = stats_summary["splits"]["train"]["category_counts"]
    dev_cats = stats_summary["splits"]["dev"]["category_counts"]

    rows = []
    for cat, count in sorted(train_cats.items(), key=lambda x: x[1], reverse=True):
        rows.append({"category": cat, "train_count": count, "dev_count": dev_cats.get(cat, 0)})
    df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=df, y="category", x="train_count", ax=ax, color="#2b5c8f")
    ax.set_title("GermEval 2017 Train Split (v1.4): Opinion Counts per Main Category")
    ax.set_xlabel("Number of <Opinion> Annotations in train_v1.4.xml")
    ax.set_ylabel("Main Category")
    for idx, row in df.iterrows():
        ax.text(row["train_count"] + 40, idx, f"{row['train_count']}", va="center", fontsize=8)
    ax.set_xlim(0, df["train_count"].max() * 1.12)
    plt.tight_layout()

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    return out


def plot_polarity_by_split(
    stats_summary: dict[str, Any],
    out_path: str | Path = "results/figures/dataset_polarity_by_split.png",
) -> Path:
    """Plot polarity proportions across train, dev, test_syn, and test_dia."""
    setup_plot_style()
    rows = []
    for split_name, sdata in stats_summary["splits"].items():
        total = max(1, sdata["num_opinions_total"])
        for pol in ("negative", "neutral", "positive"):
            cnt = sdata["polarity_counts"].get(pol, 0)
            rows.append(
                {
                    "split": split_name,
                    "polarity": pol,
                    "count": cnt,
                    "percentage": 100.0 * cnt / total,
                }
            )
    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.barplot(data=df, x="split", y="percentage", hue="polarity", ax=ax)
    ax.set_title("Aspect Polarity Distribution Across GermEval 2017 Splits (%)")
    ax.set_ylabel("Percentage of Opinions (%)")
    ax.set_xlabel("Dataset Split")
    plt.tight_layout()

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    return out


def plot_learning_curve(
    curve_df: pd.DataFrame,
    prompt_baselines: dict[str, float],
    out_path: str | Path = "results/figures/learning_curve_task_c2.png",
) -> Path:
    """Plot Task C2 Micro-F1 vs training subset size N (uniform vs stratified) with prompting baselines."""
    setup_plot_style()
    fig, ax = plt.subplots(figsize=(9, 5.2))
    sns.lineplot(
        data=curve_df,
        x="train_examples",
        y="task_c2_micro_f1",
        hue="sampling_strategy",
        style="sampling_strategy",
        markers=True,
        dashes=False,
        errorbar="sd",
        ax=ax,
    )
    colors = {"P0 (Minimal 0-shot)": "#888888", "P1 (Guidelines 0-shot)": "#d95f02", "P3 (10-shot Dev-Selected)": "#1b9e77"}
    for label, score in prompt_baselines.items():
        ax.axhline(
            score,
            linestyle="--",
            linewidth=1.3,
            color=colors.get(label, "#555555"),
            label=f"{label}: {score:.3f}",
        )
    ax.set_xscale("log")
    ax.set_title("Task C2 (Category+Sentiment) Micro-F1: 4-bit QLoRA Learning Curve vs. Prompting")
    ax.set_xlabel("Number of Fine-Tuning Training Documents (log scale)")
    ax.set_ylabel("Task C2 Micro-F1")
    ax.legend(loc="lower right", frameon=True)
    plt.tight_layout()

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    return out


def plot_per_category_delta_f1(
    prompt_metrics: dict[str, Any],
    qlora_metrics: dict[str, Any],
    out_path: str | Path = "results/figures/error_analysis_delta_f1.png",
) -> Path:
    """Plot per-category F1 gain (QLoRA minus P3 Few-Shot) ordered by training support."""
    setup_plot_style()
    p_cats = prompt_metrics["task_c1_category_strict"]["per_label"]
    q_cats = qlora_metrics["task_c1_category_strict"]["per_label"]

    rows = []
    for cat in sorted(q_cats.keys()):
        sup = q_cats[cat]["support"]
        pf1 = p_cats.get(cat, {}).get("f1", 0.0)
        qf1 = q_cats[cat]["f1"]
        rows.append(
            {
                "category": f"{cat} (n={sup})",
                "support": sup,
                "prompt_f1": pf1,
                "qlora_f1": qf1,
                "delta_f1": qf1 - pf1,
            }
        )
    df = pd.DataFrame(rows).sort_values("support", ascending=True)

    fig, ax = plt.subplots(figsize=(9.5, 6))
    colors = ["#2b8cbe" if d >= 0 else "#e34a33" for d in df["delta_f1"]]
    ax.barh(df["category"], df["delta_f1"], color=colors)
    ax.axvline(0.0, color="black", linewidth=0.9)
    ax.set_title("Per-Category F1 Gain: QLoRA (Q1) minus Best Few-Shot Prompting (P3)")
    ax.set_xlabel("Δ F1 (F1_QLoRA - F1_P3)")
    plt.tight_layout()

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    return out


def compute_paired_contingency_quadrants(
    canonical_docs: Sequence[CanonicalDocument],
    prompt_preds: Sequence[PredictionRecord],
    qlora_preds: Sequence[PredictionRecord],
) -> dict[str, Any]:
    """Compare exact-set document correctness between best prompt baseline and QLoRA."""
    p_map = {r.doc_id: set(r.parsed_labels) for r in prompt_preds}
    q_map = {r.doc_id: set(r.parsed_labels) for r in qlora_preds}

    both_correct = 0
    prompt_only = 0
    qlora_only = 0
    neither_correct = 0
    examples_qlora_only: list[dict[str, Any]] = []
    examples_prompt_only: list[dict[str, Any]] = []

    for doc in canonical_docs:
        gold = {f"{op.category}:{op.polarity}" for op in doc.opinions}
        p_set = p_map.get(doc.doc_id, set())
        q_set = q_map.get(doc.doc_id, set())
        p_ok = p_set == gold
        q_ok = q_set == gold

        if p_ok and q_ok:
            both_correct += 1
        elif p_ok and not q_ok:
            prompt_only += 1
            if len(examples_prompt_only) < 5:
                examples_prompt_only.append(
                    {
                        "doc_id": doc.doc_id,
                        "text": doc.text[:160],
                        "gold": sorted(gold),
                        "prompt_pred": sorted(p_set),
                        "qlora_pred": sorted(q_set),
                    }
                )
        elif q_ok and not p_ok:
            qlora_only += 1
            if len(examples_qlora_only) < 5:
                examples_qlora_only.append(
                    {
                        "doc_id": doc.doc_id,
                        "text": doc.text[:160],
                        "gold": sorted(gold),
                        "prompt_pred": sorted(p_set),
                        "qlora_pred": sorted(q_set),
                    }
                )
        else:
            neither_correct += 1

    total = max(1, len(canonical_docs))
    return {
        "total_documents": len(canonical_docs),
        "both_correct": both_correct,
        "qlora_only_correct": qlora_only,
        "prompt_only_correct": prompt_only,
        "neither_correct": neither_correct,
        "rates": {
            "both_correct": both_correct / total,
            "qlora_only_correct": qlora_only / total,
            "prompt_only_correct": prompt_only / total,
            "neither_correct": neither_correct / total,
        },
        "sample_qlora_wins": examples_qlora_only,
        "sample_prompt_wins": examples_prompt_only,
    }


def save_json(data: dict[str, Any], path: str | Path) -> Path:
    """Write a dictionary as formatted UTF-8 JSON."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return out


PARSE_STATUS_DESCRIPTIONS: dict[str, str] = {
    "VALID_EXACT": "Exact JSON object",
    "VALID_FENCED": "Markdown-fenced JSON (recoverable)",
    "INVALID_JSON": "JSON parsing error (unrecoverable)",
    "INVALID_SCHEMA": "Schema violation (missing/non-list 'labels')",
    "UNKNOWN_LABEL": "Unknown taxonomy label (outside 19 categories)",
}


def select_pedagogical_error_analysis_examples(
    canonical_docs: Sequence[CanonicalDocument],
    baseline_preds: Sequence[PredictionRecord],
    ft_preds: Sequence[PredictionRecord],
    apply_gold_typo_patches: bool = True,
) -> list[dict[str, Any]]:
    """Deterministically select 5 diagnostic examples for pedagogical error analysis.

    Document-level correctness is defined strictly as exact set match:
        correct(d) = 1[Y_hat_d == Y_d]
    where Y_d is the gold set of 'Category:polarity' pairs.

    Archetypes:
      1. Baseline error corrected (0-shot ❌ -> Fine-tuned ✅)
      2. Both models exact match (0-shot ✅ -> Fine-tuned ✅)
      3. Both models fail (0-shot ❌ -> Fine-tuned ❌)
      4. Fine-tuning introduces an error (0-shot ✅ -> Fine-tuned ❌)
      5. Output-format / schema issue fixed (Baseline non-VALID_EXACT -> Fine-tuned VALID_EXACT)

    If an archetype does not occur in the evaluation slice, returns an explicit
    'available': False entry rather than forcing an inappropriate example or failing.
    """
    from germeval_llm.convert import extract_task_c_label_sets
    from germeval_llm.evaluation import align_predictions

    aligned_b = align_predictions(canonical_docs, baseline_preds)
    aligned_q = align_predictions(canonical_docs, ft_preds)

    candidates: list[dict[str, Any]] = []
    for doc, bp, qp in zip(canonical_docs, aligned_b, aligned_q):
        if apply_gold_typo_patches:
            gold_list, _ = extract_task_c_label_sets(doc, apply_typo_patches=True)
        else:
            gold_list = sorted({f"{op.category}:{op.polarity}" for op in doc.opinions})
        gold_set = set(gold_list)
        b_set = set(bp.parsed_labels)
        q_set = set(qp.parsed_labels)

        b_schema_ok = bp.parse_status in ("VALID_EXACT", "VALID_FENCED")
        q_schema_ok = qp.parse_status in ("VALID_EXACT", "VALID_FENCED")
        b_exact = b_schema_ok and (b_set == gold_set)
        q_exact = q_schema_ok and (q_set == gold_set)

        candidates.append(
            {
                "doc": doc,
                "bp": bp,
                "qp": qp,
                "gold_list": gold_list,
                "gold_set": gold_set,
                "b_list": sorted(b_set),
                "q_list": sorted(q_set),
                "b_exact": b_exact,
                "q_exact": q_exact,
                "b_partial_tp": len(gold_set & b_set),
                "q_partial_tp": len(gold_set & q_set),
            }
        )

    used_doc_ids: set[str] = set()

    def _pick(predicate) -> dict[str, Any] | None:
        for item in candidates:
            if item["doc"].doc_id not in used_doc_ids and predicate(item):
                used_doc_ids.add(item["doc"].doc_id)
                return item
        return None

    # 1. Baseline error corrected (prefer specific non-Allgemein aspect category first, then non-empty gold)
    c1 = _pick(
        lambda x: (not x["b_exact"])
        and x["q_exact"]
        and any(not g.startswith("Allgemein:") for g in x["gold_list"])
    )
    if c1 is None:
        c1 = _pick(lambda x: (not x["b_exact"]) and x["q_exact"] and len(x["gold_set"]) > 0)
    if c1 is None:
        c1 = _pick(lambda x: (not x["b_exact"]) and x["q_exact"])

    # 2. Both models exact match (prefer non-empty gold)
    c2 = _pick(lambda x: x["b_exact"] and x["q_exact"] and len(x["gold_set"]) > 0)
    if c2 is None:
        c2 = _pick(lambda x: x["b_exact"] and x["q_exact"])

    # 3. Both models fail (prefer cases with non-empty gold)
    c3 = _pick(lambda x: (not x["b_exact"]) and (not x["q_exact"]) and len(x["gold_set"]) > 0)
    if c3 is None:
        c3 = _pick(lambda x: (not x["b_exact"]) and (not x["q_exact"]))

    # 4. Fine-tuning introduces an error (0-shot exact match, fine-tuned not exact match)
    c4 = _pick(lambda x: x["b_exact"] and (not x["q_exact"]) and len(x["gold_set"]) > 0)
    if c4 is None:
        c4 = _pick(lambda x: x["b_exact"] and (not x["q_exact"]))

    # 5. Output-format or schema issue fixed: prioritize unrecoverable parse/schema/label errors first,
    # then recoverable Markdown-fenced JSON (VALID_FENCED).
    c5 = _pick(
        lambda x: x["bp"].parse_status in ("INVALID_JSON", "INVALID_SCHEMA", "UNKNOWN_LABEL")
        and x["qp"].parse_status == "VALID_EXACT"
    )
    if c5 is None:
        c5 = _pick(
            lambda x: x["bp"].parse_status != "VALID_EXACT" and x["qp"].parse_status == "VALID_EXACT"
        )
    if c5 is None:
        c5 = _pick(lambda x: x["bp"].parse_status != "VALID_EXACT" or x["qp"].parse_status != "VALID_EXACT")

    archetype_specs = [
        ("1. Baseline error corrected", c1),
        ("2. Both models correct", c2),
        ("3. Both models fail", c3),
        ("4. Fine-tuning introduces error", c4),
        ("5. Output-format / schema problem", c5),
    ]

    rows: list[dict[str, Any]] = []
    for label, picked in archetype_specs:
        if picked is None:
            rows.append(
                {
                    "archetype": label,
                    "available": False,
                    "doc_id": "UNAVAILABLE",
                    "input_text": "No document in the current evaluation slice exhibited this pattern.",
                    "gold_labels": [],
                    "zero_shot_prediction": [],
                    "finetuned_prediction": [],
                    "zero_shot_correct": "N/A",
                    "finetuned_correct": "N/A",
                    "zero_shot_format": "N/A",
                    "finetuned_format": "N/A",
                    "zero_shot_raw": "",
                    "finetuned_raw": "",
                    "zero_shot_errors": [],
                    "finetuned_errors": [],
                }
            )
            continue

        bp: PredictionRecord = picked["bp"]
        qp: PredictionRecord = picked["qp"]
        b_status_desc = PARSE_STATUS_DESCRIPTIONS.get(bp.parse_status, bp.parse_status)
        q_status_desc = PARSE_STATUS_DESCRIPTIONS.get(qp.parse_status, qp.parse_status)

        def _fmt_correct(is_exact: bool, partial_tp: int, gold_n: int) -> str:
            if is_exact:
                return "Yes (exact match)"
            if partial_tp > 0 and gold_n > 0:
                return f"No (partial: {partial_tp}/{gold_n} gold matched)"
            return "No"

        rows.append(
            {
                "archetype": label,
                "available": True,
                "doc_id": picked["doc"].doc_id,
                "input_text": picked["doc"].text,
                "gold_labels": picked["gold_list"],
                "zero_shot_prediction": picked["b_list"],
                "finetuned_prediction": picked["q_list"],
                "zero_shot_correct": _fmt_correct(
                    picked["b_exact"], picked["b_partial_tp"], len(picked["gold_set"])
                ),
                "finetuned_correct": _fmt_correct(
                    picked["q_exact"], picked["q_partial_tp"], len(picked["gold_set"])
                ),
                "zero_shot_format": f"{bp.parse_status} ({b_status_desc})",
                "finetuned_format": f"{qp.parse_status} ({q_status_desc})",
                "zero_shot_raw": bp.raw_output,
                "finetuned_raw": qp.raw_output,
                "zero_shot_errors": list(bp.parse_errors),
                "finetuned_errors": list(qp.parse_errors),
            }
        )
    return rows


def plot_training_and_validation_loss(
    train_history: Sequence[dict[str, Any]],
    eval_history: Sequence[dict[str, Any]] | None = None,
    title: str = "Supervised Instruction Fine-Tuning: Completion Loss Across Optimization Steps",
    out_path: str | Path | None = None,
) -> plt.Figure:
    """Plot training loss and validation loss series at their respective logging steps."""
    setup_plot_style()
    fig, ax1 = plt.subplots(figsize=(8.8, 4.4))

    if train_history:
        t_steps = [int(e["step"]) for e in train_history]
        t_losses = [float(e["loss"]) for e in train_history]
        ax1.plot(
            t_steps,
            t_losses,
            marker="o",
            linewidth=2.0,
            color="#1f77b4",
            label="Training completion loss (batch avg)",
        )

    if eval_history:
        e_steps = [int(e["step"]) for e in eval_history]
        e_losses = [float(e["eval_loss"]) for e in eval_history]
        ax1.plot(
            e_steps,
            e_losses,
            marker="s",
            linestyle="--",
            linewidth=2.0,
            color="#d95f02",
            label="Validation completion loss (dev split)",
        )

    ax1.set_title(title)
    ax1.set_xlabel("Optimization Step")
    ax1.set_ylabel("Cross-Entropy Loss (Completion Tokens Only)")
    ax1.legend(loc="upper right", frameon=True)
    plt.tight_layout()

    if out_path is not None:
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out)
    return fig


def plot_before_after_metrics_comparison(
    comparison: dict[str, Any],
    method_label: str = "Fine-Tuned Adapter",
    out_path: str | Path | None = None,
    include_raw_json: bool = True,
) -> plt.Figure:
    """Plot grouped bar chart comparing Zero-Shot baseline vs. Fine-Tuned model on core Task C metrics."""
    setup_plot_style()
    b = comparison["baseline"]
    q = comparison["finetuned"]

    metrics_spec = [
        ("Unwrapped task JSON\n(raw compliance)", b["valid_exact_rate"], q["valid_exact_rate"]),
        ("Usable task output\n(tolerant parser)", b["schema_validity_rate"], q["schema_validity_rate"]),
        (
            "Category F1\n(Task C1)",
            b["task_c1_category_strict"]["micro_f1"],
            q["task_c1_category_strict"]["micro_f1"],
        ),
        (
            "Category + Sentiment F1\n(Task C2)",
            b["task_c2_category_sentiment_strict"]["micro_f1"],
            q["task_c2_category_sentiment_strict"]["micro_f1"],
        ),
    ]

    if not include_raw_json:
        metrics_spec = metrics_spec[1:]

    labels = [m[0] for m in metrics_spec]
    b_vals = [float(m[1]) for m in metrics_spec]
    q_vals = [float(m[2]) for m in metrics_spec]

    import numpy as np

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9.0, 5.4))
    rects1 = ax.bar(x - width / 2, b_vals, width, label="Zero-Shot Baseline (P1)", color="#8c96c6")
    rects2 = ax.bar(x + width / 2, q_vals, width, label=method_label, color="#2b8cbe")

    for r1, r2 in zip(rects1, rects2):
        h1 = r1.get_height()
        h2 = r2.get_height()
        delta = h2 - h1
        ax.text(
            r1.get_x() + r1.get_width() / 2.0,
            h1 + 0.015,
            f"{h1:.2f}",
            ha="center",
            va="bottom",
            fontsize=8.5,
        )
        ax.text(
            r2.get_x() + r2.get_width() / 2.0,
            h2 + 0.015,
            f"{h2:.2f}\n({delta:+.2f})",
            ha="center",
            va="bottom",
            fontsize=8.5,
            fontweight="bold",
        )

    ax.set_ylim(0, 1.16)
    ax.set_ylabel("Score / Proportion (0.0 – 1.0)")
    ax.set_title(
        f"Before vs. After Instruction Fine-Tuning (n = {comparison['num_documents']} evaluation documents)"
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.20), frameon=True, fontsize=9)
    plt.tight_layout()

    if out_path is not None:
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out)
    return fig

