"""End-to-end pipeline orchestrator for canonical conversion, dataset derivation, and CAS lab precomputation."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import random
from typing import Any
import pandas as pd
import yaml

from germeval_llm import __version__
from germeval_llm.alignment import align_targets_to_text
from germeval_llm.convert import (
    derive_task_c_example,
    derive_task_d_example,
    extract_task_c_label_sets,
    extract_task_d_targets,
    save_derived_jsonl,
)
from germeval_llm.evaluation import evaluate_task_c_predictions, evaluate_task_d_predictions
from germeval_llm.inference import save_predictions_jsonl
from germeval_llm.official_evaluation import (
    export_canonical_gold_to_official_tsv,
    export_explicit_only_gold_xml,
    export_task_c_to_official_tsv,
    export_task_d_to_official_xml,
    run_official_java_evaluator,
)
from germeval_llm.parsing import parse_task_c_output, parse_task_d_output
from germeval_llm.prompts import load_prompts_config
from germeval_llm.reporting import (
    compute_paired_contingency_quadrants,
    plot_dataset_category_distribution,
    plot_learning_curve,
    plot_per_category_delta_f1,
    plot_polarity_by_split,
    save_json,
)
from germeval_llm.schemas import CanonicalDocument, PredictionRecord, TaskDTargetItem
from germeval_llm.taxonomy import MAIN_CATEGORIES_19, POLARITIES, save_taxonomy_manifest
from germeval_llm.training import (
    compute_training_budget,
    estimate_gemma_parameter_and_memory_profile,
)
from germeval_llm.validate import audit_splits, save_validation_audit
from germeval_llm.xml_parser import (
    compute_sha256,
     parse_xml_file,
    save_canonical_jsonl,
    verify_lossless_roundtrip,
)


def run_m1_and_m2_data_pipeline(
    dataset_cfg_path: str | Path = "configs/dataset.yaml",
) -> dict[str, list[CanonicalDocument]]:
    """Execute Milestone 1 (canonical conversion + round-trip + audit) and Milestone 2 (Task C & D views)."""
    with Path(dataset_cfg_path).open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    prompts_cfg = load_prompts_config()
    save_taxonomy_manifest("data/manifests/taxonomy.json")

    split_docs: dict[str, list[CanonicalDocument]] = {}
    manifest_splits: dict[str, Any] = {}

    for split_name, scfg in cfg["splits"].items():
        xml_path = Path(scfg["xml_path"])
        docs = parse_xml_file(xml_path, source_split=split_name)
        ok, rt_errors = verify_lossless_roundtrip(xml_path, docs)
        if not ok:
            raise RuntimeError(f"Round-trip verification failed on {split_name}: {rt_errors[:5]}")

        can_path = save_canonical_jsonl(docs, scfg["canonical_path"])
        split_docs[split_name] = docs

        apply_patches = bool(scfg.get("apply_typo_patches", False))
        task_c_exs = [
            derive_task_c_example(d, apply_typo_patches=apply_patches, prompts_cfg=prompts_cfg)
            for d in docs
        ]
        task_d_exs = [
            derive_task_d_example(d, apply_typo_patches=apply_patches, prompts_cfg=prompts_cfg)
            for d in docs
        ]
        c_path = save_derived_jsonl(task_c_exs, scfg["task_c_path"])
        d_path = save_derived_jsonl(task_d_exs, scfg["task_d_path"])

        manifest_splits[split_name] = {
            "source_xml": str(xml_path),
            "source_xml_sha256": compute_sha256(xml_path),
            "canonical_jsonl": str(can_path),
            "canonical_sha256": compute_sha256(can_path),
            "task_c_jsonl": str(c_path),
            "task_c_sha256": compute_sha256(c_path),
            "task_d_jsonl": str(d_path),
            "task_d_sha256": compute_sha256(d_path),
            "num_documents": len(docs),
            "num_opinions": sum(len(d.opinions) for d in docs),
            "roundtrip_verified": ok,
            "typo_patches_applied_in_derived": apply_patches,
        }

    issues, stats_summary = audit_splits(split_docs)
    save_validation_audit(
        issues,
        stats_summary,
        audit_path="data/manifests/validation_audit.json",
        stats_path="data/manifests/statistics.json",
    )

    dataset_manifest = {
        "dataset_name": cfg["dataset_name"],
        "license": cfg["license"],
        "parser_version": __version__,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "splits": manifest_splits,
        "validation_total_issues": len(issues),
    }
    save_json(dataset_manifest, "data/manifests/dataset_manifest.json")

    # Export dataset exploration figures
    plot_dataset_category_distribution(stats_summary)
    plot_polarity_by_split(stats_summary)

    return split_docs


def _simulate_realistic_task_c_run(
    docs: list[CanonicalDocument],
    run_id: str,
    split: str,
    recall_prob: float,
    rare_recall_penalty: float,
    fp_prob_relevant: float,
    fp_prob_irrelevant: float,
    polarity_confusion_prob: float,
    fenced_rate: float,
    invalid_json_rate: float,
    prompt_tokens_base: int,
    seed: int,
) -> list[PredictionRecord]:
    """Generate deterministic, realistic per-document Task C predictions reflecting empirical error modes
    (rare category recall drop, false positives on irrelevant docs, polarity confusions, and JSON formatting)
    for precomputed CAS classroom fallback artifacts.
    """
    rng = random.Random(seed)
    frequent_cats = {"Allgemein", "Zugfahrt", "Sonstige_Unregelmässigkeiten", "Service_und_Kundenbetreuung", "Sicherheit"}
    all_cats = list(MAIN_CATEGORIES_19)
    all_pols = list(POLARITIES)

    records: list[PredictionRecord] = []
    for doc in docs:
        gold_labels, _ = extract_task_c_label_sets(doc, apply_typo_patches=True)
        p_toks = prompt_tokens_base + max(8, len(doc.text) // 4)

        r_val = rng.random()
        if r_val < invalid_json_rate:
            raw_out = '{"labels": ["Allgemein:negative"'  # truncated malformed JSON
            parsed_labels, status, errors = parse_task_c_output(raw_out)
            records.append(
                PredictionRecord(
                    run_id=run_id,
                    task="C",
                    split=split,
                    doc_id=doc.doc_id,
                    source_ordinal=doc.source_ordinal,
                    raw_output=raw_out,
                    parse_status=status,
                    parse_errors=errors,
                    parsed_labels=parsed_labels,
                    prompt_tokens=p_toks,
                    completion_tokens=12,
                    latency_ms=round(18.0 + p_toks * 0.04, 2),
                )
            )
            continue

        pred_set: set[str] = set()
        for glbl in gold_labels:
            cat, pol = glbl.split(":", 1)
            eff_recall = recall_prob if cat in frequent_cats else max(0.05, recall_prob - rare_recall_penalty)
            if rng.random() < eff_recall:
                if rng.random() < polarity_confusion_prob:
                    alt_pols = [p for p in all_pols if p != pol]
                    pol = rng.choice(alt_pols)
                pred_set.add(f"{cat}:{pol}")

        # False positive injection
        if len(gold_labels) == 0:
            if rng.random() < fp_prob_irrelevant:
                pred_set.add(f"Allgemein:{rng.choice(['neutral', 'negative'])}")
        else:
            if rng.random() < fp_prob_relevant:
                fp_cat = rng.choice(["Allgemein", "Zugfahrt", "Image", rng.choice(all_cats)])
                fp_pol = doc.document_sentiment if doc.document_sentiment in all_pols else "negative"
                pred_set.add(f"{fp_cat}:{fp_pol}")

        sorted_pred = sorted(pred_set)
        json_str = json.dumps({"labels": sorted_pred}, ensure_ascii=False)
        if rng.random() < fenced_rate:
            raw_out = f"```json\n{json_str}\n```"
        else:
            raw_out = json_str

        parsed_labels, status, errors = parse_task_c_output(raw_out)
        c_toks = max(6, len(raw_out) // 3)
        records.append(
            PredictionRecord(
                run_id=run_id,
                task="C",
                split=split,
                doc_id=doc.doc_id,
                source_ordinal=doc.source_ordinal,
                raw_output=raw_out,
                parse_status=status,
                parse_errors=errors,
                parsed_labels=parsed_labels,
                prompt_tokens=p_toks,
                completion_tokens=c_toks,
                latency_ms=round(14.0 + p_toks * 0.035, 2),
            )
        )
    return records


def _simulate_realistic_task_d_run(
    docs: list[CanonicalDocument],
    run_id: str,
    split: str,
    contiguous_recall: float,
    star_recall: float,
    spurious_prob: float,
    prompt_tokens_base: int,
    seed: int,
) -> list[PredictionRecord]:
    """Generate deterministic Task D target-string predictions and run real deterministic span alignment."""
    rng = random.Random(seed)
    records: list[PredictionRecord] = []

    for doc in docs:
        gold_targets = extract_task_d_targets(doc, apply_typo_patches=True, view="exact_and_star")
        pred_items: list[TaskDTargetItem] = []

        for gt in gold_targets:
            rec_p = star_recall if "*" in gt.target else contiguous_recall
            if rng.random() < rec_p:
                pred_items.append(gt)
            elif "*" in gt.target and rng.random() < 0.45:
                # Realistic error: model extracts only the first part of a discontinuous target
                first_part = gt.target.split("*")[0].strip()
                if first_part:
                    pred_items.append(TaskDTargetItem(target=first_part, category=gt.category, polarity=gt.polarity))

        if rng.random() < spurious_prob:
            words = [w.strip(".,!?:;\"'()") for w in doc.text.split() if len(w.strip(".,!?:;\"'()")) >= 4]
            if words:
                sp_word = rng.choice(words)
                pred_items.append(TaskDTargetItem(target=sp_word, category="Allgemein", polarity="neutral"))

        raw_out = json.dumps({"targets": [t.model_dump() for t in pred_items]}, ensure_ascii=False)
        parsed_targets, status, errors = parse_task_d_output(raw_out)
        aligned_spans = align_targets_to_text(doc.text, parsed_targets)
        p_toks = prompt_tokens_base + max(8, len(doc.text) // 4)

        records.append(
            PredictionRecord(
                run_id=run_id,
                task="D",
                split=split,
                doc_id=doc.doc_id,
                source_ordinal=doc.source_ordinal,
                raw_output=raw_out,
                parse_status=status,
                parse_errors=errors,
                parsed_targets=parsed_targets,
                aligned_spans=aligned_spans,
                prompt_tokens=p_toks,
                completion_tokens=max(8, len(raw_out) // 3),
                latency_ms=round(18.0 + p_toks * 0.04, 2),
            )
        )
    return records


def run_m3_to_m6_benchmark_artifacts(
    split_docs: dict[str, list[CanonicalDocument]],
) -> dict[str, Any]:
    """Generate simulated teaching artifacts exclusively under results/simulations/.

    Scores and hardware profiles are synthetic, not measured model executions.
    """
    save_json(
        {"execution_mode": "simulation", "split_seed_offsets": {"dev": 0, "test_syn": 1, "test_dia": 2}},
        "results/simulations/provenance.json",
    )
    task_c_specs = {
        "P0_minimal_0shot": {
            "method": "prompt_zero_shot_minimal",
            "recall": 0.49,
            "rare_pen": 0.25,
            "fp_rel": 0.22,
            "fp_irrel": 0.34,
            "pol_conf": 0.16,
            "fenced": 0.28,
            "inv_json": 0.018,
            "p_toks": 165,
            "seed": 101,
        },
        "P1_guidelines_0shot": {
            "method": "prompt_zero_shot_guidelines",
            "recall": 0.58,
            "rare_pen": 0.18,
            "fp_rel": 0.16,
            "fp_irrel": 0.21,
            "pol_conf": 0.12,
            "fenced": 0.09,
            "inv_json": 0.005,
            "p_toks": 540,
            "seed": 102,
        },
        "P2_fewshot_2": {
            "method": "prompt_few_shot_2",
            "recall": 0.62,
            "rare_pen": 0.16,
            "fp_rel": 0.14,
            "fp_irrel": 0.15,
            "pol_conf": 0.10,
            "fenced": 0.02,
            "inv_json": 0.002,
            "p_toks": 760,
            "seed": 103,
        },
        "P2_fewshot_5": {
            "method": "prompt_few_shot_5",
            "recall": 0.65,
            "rare_pen": 0.14,
            "fp_rel": 0.12,
            "fp_irrel": 0.11,
            "pol_conf": 0.09,
            "fenced": 0.01,
            "inv_json": 0.001,
            "p_toks": 1080,
            "seed": 104,
        },
        "P2_fewshot_10": {
            "method": "prompt_few_shot_10",
            "recall": 0.67,
            "rare_pen": 0.13,
            "fp_rel": 0.11,
            "fp_irrel": 0.09,
            "pol_conf": 0.085,
            "fenced": 0.005,
            "inv_json": 0.0,
            "p_toks": 1620,
            "seed": 105,
        },
        "P3_dev_optimized_10shot": {
            "method": "prompt_dev_selected_10shot",
            "recall": 0.695,
            "rare_pen": 0.11,
            "fp_rel": 0.095,
            "fp_irrel": 0.075,
            "pol_conf": 0.075,
            "fenced": 0.002,
            "inv_json": 0.0,
            "p_toks": 1640,
            "seed": 106,
        },
        "L1_lora_r16": {
            "method": "lora_bf16_r16",
            "recall": 0.815,
            "rare_pen": 0.045,
            "fp_rel": 0.055,
            "fp_irrel": 0.025,
            "pol_conf": 0.042,
            "fenced": 0.0,
            "inv_json": 0.0,
            "p_toks": 540,
            "seed": 201,
        },
        "Q1_qlora_r16": {
            "method": "qlora_nf4_r16",
            "recall": 0.810,
            "rare_pen": 0.048,
            "fp_rel": 0.058,
            "fp_irrel": 0.027,
            "pol_conf": 0.044,
            "fenced": 0.0,
            "inv_json": 0.0,
            "p_toks": 540,
            "seed": 202,
        },
    }

    benchmark_rows: list[dict[str, Any]] = []
    eval_splits = ["dev", "test_syn", "test_dia"]

    for run_id, spec in task_c_specs.items():
        run_metrics_by_split: dict[str, Any] = {}
        for sp in eval_splits:
            diachronic_drop = 0.025 if sp == "test_dia" else 0.0
            preds = _simulate_realistic_task_c_run(
                split_docs[sp],
                run_id=run_id,
                split=sp,
                recall_prob=spec["recall"] - diachronic_drop,
                rare_recall_penalty=spec["rare_pen"],
                fp_prob_relevant=spec["fp_rel"],
                fp_prob_irrelevant=spec["fp_irrel"],
                polarity_confusion_prob=spec["pol_conf"],
                fenced_rate=spec["fenced"],
                invalid_json_rate=spec["inv_json"],
                prompt_tokens_base=spec["p_toks"],
                seed=spec["seed"] + {"dev": 0, "test_syn": 1, "test_dia": 2}[sp],
            )
            save_predictions_jsonl(preds, f"results/simulations/predictions/{run_id}_{sp}.jsonl")
            m = evaluate_task_c_predictions(split_docs[sp], preds, n_bootstrap=200)
            m["execution_mode"] = "simulation"
            run_metrics_by_split[sp] = m

            benchmark_rows.append(
                {
                    "execution_mode": "simulation",
                    "run_id": run_id,
                    "method": spec["method"],
                    "split": sp,
                    "c1_strict_micro_f1": round(m["task_c1_category_strict"]["micro_f1"], 4),
                    "c2_strict_micro_f1": round(m["task_c2_category_sentiment_strict"]["micro_f1"], 4),
                    "c2_strict_macro_f1": round(m["task_c2_category_sentiment_strict"]["macro_f1_active"], 4),
                    "c1_java_compat_micro_f1": round(m["task_c1_category_java_compat"]["micro_f1"], 4),
                    "c2_java_compat_micro_f1": round(m["task_c2_category_sentiment_java_compat"]["micro_f1"], 4),
                    "exact_set_accuracy": round(m["task_c2_category_sentiment_strict"]["exact_set_accuracy"], 4),
                    "schema_validity_rate": round(m["schema_validity_rate"], 4),
                    "prompt_tokens_per_1k_docs": round(m["efficiency"]["prompt_tokens_per_1k_docs"], 1),
                }
            )

        save_json(run_metrics_by_split, f"results/simulations/metrics/{run_id}.json")

        # Save run manifest
        is_lora = "lora" in spec["method"]
        quant = "nf4_4bit" if "qlora" in spec["method"] else "bf16_16bit"
        hw_profile = (
            estimate_gemma_parameter_and_memory_profile(lora_rank=16, quantization=quant)
            if is_lora
            else {"model_id": "google/gemma-2-2b-it", "trainable_parameters": 0, "estimated_peak_vram_mb": 2650.0}
        )
        save_json(
            {
                "execution_mode": "simulation",
                "run_id": run_id,
                "task": "C",
                "method": spec["method"],
                "base_model_id": "google/gemma-2-2b-it",
                "prompt_version": "v1",
                "taxonomy_version": "v1_19cats",
                "seed": spec["seed"],
                "hardware_and_parameter_profile": hw_profile,
                "summary_dev_c2_micro_f1": run_metrics_by_split["dev"]["task_c2_category_sentiment_strict"][
                    "micro_f1"
                ],
                "summary_test_syn_c2_micro_f1": run_metrics_by_split["test_syn"][
                    "task_c2_category_sentiment_strict"
                ]["micro_f1"],
                "summary_test_dia_c2_micro_f1": run_metrics_by_split["test_dia"][
                    "task_c2_category_sentiment_strict"
                ]["micro_f1"],
            },
            f"results/simulations/runs/{run_id}.json",
        )

    # Official EvaluationScript.jar reconciliation on test_syn for Q1_qlora_r16 and P3_dev_optimized_10shot
    reconciliation_records: list[dict[str, Any]] = []
    for run_id in ("P3_dev_optimized_10shot", "Q1_qlora_r16"):
        from germeval_llm.inference import load_predictions_jsonl

        preds_syn = load_predictions_jsonl(f"results/simulations/predictions/{run_id}_test_syn.jsonl")
        tsv_pred_path = export_task_c_to_official_tsv(
            split_docs["test_syn"], preds_syn, f"results/simulations/predictions/{run_id}_test_syn_official.tsv"
        )
        java_res = run_official_java_evaluator(
            "category",
            predicted_file=tsv_pred_path,
            gold_file="data/raw/test_TIMESTAMP1.tsv",
        )
        py_res = json.loads(Path(f"results/simulations/metrics/{run_id}.json").read_text(encoding="utf-8"))["test_syn"]
        reconciliation_records.append(
            {
                "execution_mode": "simulation",
                "run_id": run_id,
                "split": "test_syn",
                "python_strict_c1_micro_f1": round(py_res["task_c1_category_strict"]["micro_f1"], 6),
                "python_java_compat_c1_micro_f1": round(py_res["task_c1_category_java_compat"]["micro_f1"], 6),
                "official_jar_c1_micro_f1": round(java_res["task_c1_category_micro_f1"], 6),
                "python_strict_c2_micro_f1": round(py_res["task_c2_category_sentiment_strict"]["micro_f1"], 6),
                "python_java_compat_c2_micro_f1": round(
                    py_res["task_c2_category_sentiment_java_compat"]["micro_f1"], 6
                ),
                "official_jar_c2_micro_f1": round(java_res["task_c2_category_sentiment_micro_f1"], 6),
                "exact_agreement_c1": abs(
                    py_res["task_c1_category_java_compat"]["micro_f1"] - java_res["task_c1_category_micro_f1"]
                )
                < 1e-6,
                "exact_agreement_c2": abs(
                    py_res["task_c2_category_sentiment_java_compat"]["micro_f1"]
                    - java_res["task_c2_category_sentiment_micro_f1"]
                )
                < 1e-6,
            }
        )
    save_json({"reconciliations": reconciliation_records}, "results/simulations/metrics/official_java_reconciliation.json")

    # Q2 Learning curves (subset sizes 100..19432, uniform vs stratified, 3 seeds)
    lc_rows: list[dict[str, Any]] = []
    subset_sizes = [100, 250, 500, 1000, 2000, 4000, 19432]
    for n_ex in subset_sizes:
        budget = compute_training_budget(n_ex, effective_batch_size=16, target_epochs=3, min_optimizer_steps_floor=100)
        for strat in ("uniform", "stratified"):
            for seed in (42, 100, 2026):
                rng = random.Random(seed + n_ex + (0 if strat == "uniform" else 500))
                import math

                prog = (math.log10(n_ex) - math.log10(100)) / (math.log10(19432) - math.log10(100))
                strat_bonus = 0.024 * (1.0 - prog) if strat == "stratified" else 0.0
                c2_f1 = 0.565 + 0.215 * (prog**0.68) + strat_bonus + rng.gauss(0, 0.004)
                c2_macro = 0.29 + 0.34 * (prog**0.75) + (strat_bonus * 1.6) + rng.gauss(0, 0.006)
                lc_rows.append(
                    {
                        "execution_mode": "simulation",
                        "train_examples": n_ex,
                        "sampling_strategy": strat,
                        "seed": seed,
                        "optimizer_steps": budget["max_steps"],
                        "effective_epochs": budget["effective_epochs"],
                        "task_c2_micro_f1": round(min(0.79, c2_f1), 4),
                        "task_c2_macro_f1": round(min(0.66, c2_macro), 4),
                        "gpu_minutes": round(budget["max_steps"] * 0.0085, 2),
                    }
                )
    lc_df = pd.DataFrame(lc_rows)
    lc_df.to_csv("results/simulations/metrics/learning_curves_q2.csv", index=False)

    dev_p0 = json.loads(Path("results/simulations/metrics/P0_minimal_0shot.json").read_text())["dev"][
        "task_c2_category_sentiment_strict"
    ]["micro_f1"]
    dev_p1 = json.loads(Path("results/simulations/metrics/P1_guidelines_0shot.json").read_text())["dev"][
        "task_c2_category_sentiment_strict"
    ]["micro_f1"]
    dev_p3 = json.loads(Path("results/simulations/metrics/P3_dev_optimized_10shot.json").read_text())["dev"][
        "task_c2_category_sentiment_strict"
    ]["micro_f1"]
    plot_learning_curve(
        lc_df,
        out_path="results/simulations/figures/learning_curve_task_c2.png",
        prompt_baselines={
            "P0 (Minimal 0-shot)": dev_p0,
            "P1 (Guidelines 0-shot)": dev_p1,
            "P3 (10-shot Dev-Selected)": dev_p3,
        },
    )

    # Task D extraction runs (D1 prompting vs D2 QLoRA on dev and test_syn)
    task_d_specs = {
        "D1_prompt_5shot": {"contig_rec": 0.54, "star_rec": 0.14, "spurious": 0.18, "p_toks": 980, "seed": 301},
        "D2_qlora_r16": {"contig_rec": 0.74, "star_rec": 0.52, "spurious": 0.07, "p_toks": 490, "seed": 302},
    }
    task_d_summary: dict[str, Any] = {}
    for d_run_id, d_spec in task_d_specs.items():
        task_d_summary[d_run_id] = {}
        for sp in ("dev", "test_syn"):
            d_preds = _simulate_realistic_task_d_run(
                split_docs[sp],
                run_id=d_run_id,
                split=sp,
                contiguous_recall=d_spec["contig_rec"],
                star_recall=d_spec["star_rec"],
                spurious_prob=d_spec["spurious"],
                prompt_tokens_base=d_spec["p_toks"],
                seed=d_spec["seed"] + {"dev": 0, "test_syn": 1, "test_dia": 2}[sp],
            )
            save_predictions_jsonl(d_preds, f"results/simulations/predictions/{d_run_id}_{sp}.jsonl")
            d_metrics = evaluate_task_d_predictions(split_docs[sp], d_preds)
            d_metrics["execution_mode"] = "simulation"

            # Official XML OTE evaluation against explicit-only gold XML and raw all-opinions gold XML
            pred_xml = export_task_d_to_official_xml(
                split_docs[sp], d_preds, f"results/simulations/predictions/{d_run_id}_{sp}_ote.xml"
            )
            explicit_gold_xml = export_explicit_only_gold_xml(
                split_docs[sp], f"results/simulations/predictions/gold_{sp}_explicit_only.xml"
            )
            java_ote_explicit = run_official_java_evaluator("OTE", pred_xml, explicit_gold_xml)
            d_metrics["official_jar_explicit_gold_ote"] = {
                "exact_micro_f1": java_ote_explicit["ote_exact_micro_f1"],
                "overlap_micro_f1": java_ote_explicit["ote_overlap_micro_f1"],
            }
            task_d_summary[d_run_id][sp] = d_metrics

    save_json(task_d_summary, "results/simulations/metrics/task_d_summary.json")

    # Error analysis plots & paired quadrant analysis on test_syn
    from germeval_llm.inference import load_predictions_jsonl

    p3_metrics_syn = json.loads(Path("results/simulations/metrics/P3_dev_optimized_10shot.json").read_text())["test_syn"]
    q1_metrics_syn = json.loads(Path("results/simulations/metrics/Q1_qlora_r16.json").read_text())["test_syn"]
    plot_per_category_delta_f1(
        p3_metrics_syn, q1_metrics_syn,
        out_path="results/simulations/figures/error_analysis_delta_f1.png",
    )

    p3_preds_syn = load_predictions_jsonl("results/simulations/predictions/P3_dev_optimized_10shot_test_syn.jsonl")
    q1_preds_syn = load_predictions_jsonl("results/simulations/predictions/Q1_qlora_r16_test_syn.jsonl")
    quadrants = compute_paired_contingency_quadrants(split_docs["test_syn"], p3_preds_syn, q1_preds_syn)
    save_json(quadrants, "results/simulations/metrics/error_analysis_quadrants_test_syn.json")

    bench_df = pd.DataFrame(benchmark_rows)
    bench_df.to_csv("results/simulations/metrics/benchmark.csv", index=False)
    return {
        "benchmark_rows": len(benchmark_rows),
        "reconciliations": reconciliation_records,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Prepare data; optionally generate isolated simulated benchmarks")
    parser.add_argument("--simulate", action="store_true", help="Also write synthetic benchmarks under results/simulations/")
    args = parser.parse_args()
    splits = run_m1_and_m2_data_pipeline()
    if args.simulate:
        run_m3_to_m6_benchmark_artifacts(splits)
    print("Pipeline completed successfully.")
