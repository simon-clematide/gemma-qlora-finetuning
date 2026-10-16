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
