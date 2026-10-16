"""Command-line interface for running GermEval 2017 Gemma prompting, LoRA/QLoRA training, and evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from germeval_llm.convert import ensure_derived_datasets_ready, select_few_shot_examples_task_c
from germeval_llm.evaluation import evaluate_task_c_predictions, evaluate_task_d_predictions
from germeval_llm.inference import (
    build_hf_generator_fn,
    load_gemma_model_and_tokenizer,
    load_predictions_jsonl,
    run_task_c_inference_with_generator,
    run_task_d_inference_with_generator,
    save_predictions_jsonl,
)
from germeval_llm.official_evaluation import (
    export_task_c_to_official_tsv,
    run_official_java_evaluator,
)
from germeval_llm.reporting import save_json
from germeval_llm.schemas import TaskCExample
from germeval_llm.training import run_sft_adapter_training
from germeval_llm.xml_parser import load_canonical_jsonl


def _cmd_prompt(args: argparse.Namespace) -> None:
    docs = load_canonical_jsonl(f"data/canonical/{args.split}.jsonl")
    if args.max_docs:
        docs = docs[: args.max_docs]

    few_shots: list[TaskCExample] | None = None
    if args.n_shots > 0:
        if not Path("data/derived/task_c/train.jsonl").exists():
            ensure_derived_datasets_ready()
        train_exs: list[TaskCExample] = []
        with Path("data/derived/task_c/train.jsonl").open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    train_exs.append(TaskCExample.model_validate_json(line))
        few_shots = select_few_shot_examples_task_c(
            train_exs,
            n_shots=args.n_shots,
            strategy=args.shot_strategy,
            seed=args.seed,
        )

    model, tokenizer = load_gemma_model_and_tokenizer(
        model_id=args.model_id,
        adapter_path=args.adapter_path,
        load_in_4bit=args.load_in_4bit,
    )
    gen_fn = build_hf_generator_fn(model, tokenizer, max_new_tokens=args.max_new_tokens)

    if args.task == "C":
        preds = run_task_c_inference_with_generator(
            docs,
            run_id=args.run_id,
            split=args.split,
            generate_fn=gen_fn,
            prompt_key=args.prompt_key,
            few_shot_examples=few_shots,
        )
        out_pred = save_predictions_jsonl(preds, f"results/predictions/{args.run_id}_{args.split}.jsonl")
        metrics = evaluate_task_c_predictions(docs, preds)
    else:
        preds = run_task_d_inference_with_generator(
            docs,
            run_id=args.run_id,
            split=args.split,
            generate_fn=gen_fn,
            prompt_key=args.prompt_key,
        )
        out_pred = save_predictions_jsonl(preds, f"results/predictions/{args.run_id}_{args.split}.jsonl")
        metrics = evaluate_task_d_predictions(docs, preds)

    out_metrics = save_json(metrics, f"results/metrics/{args.run_id}_{args.split}.json")
    print(f"Saved predictions to {out_pred} and metrics to {out_metrics}")


def _cmd_train(args: argparse.Namespace) -> None:
    train_jsonl = f"data/derived/task_{args.task.lower()}/train.jsonl"
    dev_jsonl = f"data/derived/task_{args.task.lower()}/dev.jsonl"
    summary = run_sft_adapter_training(
        config_path=args.config,
        train_jsonl_path=train_jsonl,
        dev_jsonl_path=dev_jsonl,
        output_adapter_dir=args.output_dir,
        max_steps_override=args.max_steps,
        subset_size=args.subset_size,
        sampling_strategy=args.sampling_strategy,
        seed_override=args.seed,
    )
    save_json(summary, f"results/runs/{Path(args.output_dir).name}_train_summary.json")
    print(json.dumps(summary, indent=2))


def _cmd_evaluate(args: argparse.Namespace) -> None:
    docs = load_canonical_jsonl(f"data/canonical/{args.split}.jsonl")
    preds = load_predictions_jsonl(args.predictions)
    if args.task == "C":
        metrics = evaluate_task_c_predictions(docs, preds)
        if args.run_official_jar and args.split in ("test_syn", "test_dia"):
            gold_tsv = "data/raw/test_TIMESTAMP1.tsv" if args.split == "test_syn" else "data/raw/test_TIMESTAMP2.tsv"
            tmp_tsv = export_task_c_to_official_tsv(docs, preds, f"{args.predictions}.official.tsv")
            metrics["official_jar"] = run_official_java_evaluator("category", tmp_tsv, gold_tsv)
    else:
        metrics = evaluate_task_d_predictions(docs, preds)

    print(json.dumps(metrics, indent=2, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="GermEval 2017 Gemma Prompting, LoRA & QLoRA CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    p_prompt = sub.add_parser("infer", help="Run Gemma prompting or adapter inference on a split")
    p_prompt.add_argument("--run-id", required=True)
    p_prompt.add_argument("--task", choices=["C", "D"], default="C")
    p_prompt.add_argument("--split", choices=["dev", "test_syn", "test_dia"], default="dev")
    p_prompt.add_argument("--model-id", default="google/gemma-2-2b-it")
    p_prompt.add_argument("--adapter-path", default=None)
    p_prompt.add_argument("--load-in-4bit", action="store_true")
    p_prompt.add_argument("--prompt-key", default="p1_guidelines")
    p_prompt.add_argument("--n-shots", type=int, default=0)
    p_prompt.add_argument("--shot-strategy", choices=["uniform", "stratified_diverse"], default="stratified_diverse")
    p_prompt.add_argument("--max-docs", type=int, default=None)
    p_prompt.add_argument("--max-new-tokens", type=int, default=256)
    p_prompt.add_argument("--seed", type=int, default=42)
    p_prompt.set_defaults(func=_cmd_prompt)

    p_train = sub.add_parser("train", help="Train a LoRA or 4-bit NF4 QLoRA adapter with TRL SFTTrainer")
    p_train.add_argument("--config", required=True, help="Path to configs/lora.yaml or configs/qlora.yaml")
    p_train.add_argument("--task", choices=["C", "D"], default="C")
    p_train.add_argument("--output-dir", required=True, help="Output directory under models/adapters/")
    p_train.add_argument("--max-steps", type=int, default=None, help="Optional step cap (e.g. 100 for classroom smoke)")
    p_train.add_argument("--subset-size", type=int, default=None, help="Optional training subset size N")
    p_train.add_argument("--sampling-strategy", choices=["uniform", "stratified"], default="stratified")
    p_train.add_argument("--seed", type=int, default=42)
    p_train.set_defaults(func=_cmd_train)

    p_eval = sub.add_parser("evaluate", help="Evaluate saved JSONL predictions against canonical gold and Java scorer")
    p_eval.add_argument("--predictions", required=True)
    p_eval.add_argument("--task", choices=["C", "D"], default="C")
    p_eval.add_argument("--split", choices=["dev", "test_syn", "test_dia"], default="test_syn")
    p_eval.add_argument("--run-official-jar", action="store_true")
    p_eval.set_defaults(func=_cmd_evaluate)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
