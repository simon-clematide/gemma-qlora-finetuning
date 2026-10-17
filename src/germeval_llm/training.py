"""LoRA and 4-bit NF4 QLoRA configuration and TRL SFTTrainer orchestration for Gemma Instruct."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any
import yaml


GEMMA_LINEAR_MODULES: list[str] = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


def load_training_config(config_path: str | Path) -> dict[str, Any]:
    """Load LoRA or QLoRA YAML configuration."""
    with Path(config_path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def compute_training_budget(
    num_examples: int,
    effective_batch_size: int = 16,
    target_epochs: int = 3,
    min_optimizer_steps_floor: int = 100,
) -> dict[str, int | float]:
    """Compute optimizer step budget for a dataset subset of size N.

    Enforces `min_optimizer_steps_floor` so small subsets (e.g., N=100 -> 6 steps/epoch) receive
    at least 100 gradient updates rather than terminating after 18 steps during learning-curve runs.
    """
    steps_per_epoch = max(1, math.ceil(num_examples / effective_batch_size))
    nominal_steps = steps_per_epoch * target_epochs
    max_steps = max(nominal_steps, min_optimizer_steps_floor)
    effective_epochs = round(max_steps / steps_per_epoch, 2)
    return {
        "num_examples": num_examples,
        "effective_batch_size": effective_batch_size,
        "steps_per_epoch": steps_per_epoch,
        "nominal_epochs": target_epochs,
        "nominal_steps": nominal_steps,
        "max_steps": max_steps,
        "effective_epochs": effective_epochs,
    }


def estimate_gemma_parameter_and_memory_profile(
    base_params: int = 2_614_341_888,
    hidden_size: int = 2304,
    intermediate_size: int = 9216,
    num_layers: int = 26,
    num_heads: int = 8,
    num_kv_heads: int = 4,
    head_dim: int = 256,
    lora_rank: int = 16,
    quantization: str = "nf4_4bit",
) -> dict[str, Any]:
    """Analytical parameter and VRAM profile for `google/gemma-2-2b-it` under 16-bit LoRA vs 4-bit QLoRA."""
    # Per layer linear projection dimensions in Gemma-2-2B:
    # q_proj: 2304 -> 2048; k_proj: 2304 -> 1024; v_proj: 2304 -> 1024; o_proj: 2048 -> 2304
    # gate_proj: 2304 -> 9216; up_proj: 2304 -> 9216; down_proj: 9216 -> 2304
    q_dim = num_heads * head_dim
    kv_dim = num_kv_heads * head_dim

    per_layer_lora_params = lora_rank * (
        (hidden_size + q_dim)  # q_proj
        + (hidden_size + kv_dim)  # k_proj
        + (hidden_size + kv_dim)  # v_proj
        + (q_dim + hidden_size)  # o_proj
        + (hidden_size + intermediate_size)  # gate_proj
        + (hidden_size + intermediate_size)  # up_proj
        + (intermediate_size + hidden_size)  # down_proj
    )
    trainable_params = num_layers * per_layer_lora_params
    total_params = base_params + trainable_params
    trainable_pct = 100.0 * trainable_params / total_params

    if quantization == "nf4_4bit":
        # 4.5 bits/param with double quantization on non-embedding layers + bf16 embeddings/norm
        base_weights_mb = round((base_params * 0.5625) / (1024**2), 1)
    else:
        # 2 bytes/param in bf16
        base_weights_mb = round((base_params * 2.0) / (1024**2), 1)

    adapter_weights_mb = round((trainable_params * 2.0) / (1024**2), 2)
    optimizer_states_mb = round((trainable_params * 2.0) / (1024**2), 2)  # 8-bit AdamW states
    activations_grad_ckpt_mb = 1450.0 if quantization == "nf4_4bit" else 2100.0
    estimated_peak_vram_mb = round(
        base_weights_mb + adapter_weights_mb + optimizer_states_mb + activations_grad_ckpt_mb, 1
    )

    return {
        "model_id": "google/gemma-2-2b-it",
        "quantization": quantization,
        "lora_rank": lora_rank,
        "base_parameters": base_params,
        "trainable_parameters": trainable_params,
        "total_parameters": total_params,
        "trainable_percent": round(trainable_pct, 4),
        "base_weights_mb": base_weights_mb,
        "adapter_checkpoint_mb": adapter_weights_mb,
        "estimated_peak_vram_mb": estimated_peak_vram_mb,
    }


def build_peft_lora_config(cfg: dict[str, Any]) -> Any:
    """Instantiate a `peft.LoraConfig` from a YAML config dictionary."""
    from peft import LoraConfig

    p = cfg["peft"]
    return LoraConfig(
        r=int(p["r"]),
        lora_alpha=int(p["lora_alpha"]),
        lora_dropout=float(p["lora_dropout"]),
        bias=str(p.get("bias", "none")),
        task_type=str(p.get("task_type", "CAUSAL_LM")),
        target_modules=list(p.get("target_modules", GEMMA_LINEAR_MODULES)),
    )


def run_sft_adapter_training(
    config_path: str | Path,
    train_jsonl_path: str | Path,
    dev_jsonl_path: str | Path,
    output_adapter_dir: str | Path,
    max_steps_override: int | None = None,
    subset_size: int | None = None,
    sampling_strategy: str = "stratified",
    seed_override: int | None = None,
) -> dict[str, Any]:
    """Execute supervised fine-tuning (LoRA or 4-bit NF4 QLoRA) on Gemma Instruct with completion-only loss masking.

    Uses TRL `SFTConfig(completion_only_loss=True)` with explicit `prompt` and `completion` dataset columns so
    100% of prompt tokens up to `<start_of_turn>model\\n` are masked with `-100`.
    """
    import json
    import time
    import torch
    from datasets import Dataset
    from peft import get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    from germeval_llm.convert import ensure_derived_datasets_ready, sample_learning_curve_subset
    from germeval_llm.inference import detect_compute_device
    from germeval_llm.prompts import format_gemma_chat_string
    from germeval_llm.schemas import ChatMessage, TaskCExample

    cfg = load_training_config(config_path)
    tcfg = cfg["training"]
    seed = int(seed_override if seed_override is not None else tcfg.get("seed", 42))
    torch.manual_seed(seed)

    if not Path(train_jsonl_path).exists() or not Path(dev_jsonl_path).exists():
        ensure_derived_datasets_ready()

    # Load derived JSONL examples (supports both Task C and Task D JSONL)
    raw_train_rows: list[dict[str, Any]] = []
    with Path(train_jsonl_path).open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                raw_train_rows.append(json.loads(line))

    if subset_size is not None and subset_size < len(raw_train_rows):
        if "gold_labels" in raw_train_rows[0]:
            c_exs = [TaskCExample.model_validate(r) for r in raw_train_rows]
            sampled = sample_learning_curve_subset(
                c_exs,
                subset_size=subset_size,
                strategy="stratified" if sampling_strategy == "stratified" else "uniform",
                seed=seed,
            )
            raw_train_rows = [ex.model_dump() for ex in sampled]
        else:
            raw_train_rows = raw_train_rows[:subset_size]

    raw_dev_rows: list[dict[str, Any]] = []
    with Path(dev_jsonl_path).open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                raw_dev_rows.append(json.loads(line))

    def _to_prompt_completion_dataset(rows: list[dict[str, Any]]) -> Dataset:
        prompts: list[str] = []
        completions: list[str] = []
        for r in rows:
            p_msgs = [ChatMessage.model_validate(m) for m in r["prompt"]]
            c_msgs = [ChatMessage.model_validate(m) for m in r["completion"]]
            p_str = format_gemma_chat_string(p_msgs, completion_messages=None, include_bos=False)
            c_str = f"{c_msgs[0].content}<end_of_turn>\n"
            prompts.append(p_str)
            completions.append(c_str)
        return Dataset.from_dict({"prompt": prompts, "completion": completions})

    hf_train = _to_prompt_completion_dataset(raw_train_rows)
    hf_dev = _to_prompt_completion_dataset(raw_dev_rows[:250])

    model_id = cfg["base_model_id"]
    hw = detect_compute_device()
    compute_dtype = torch.bfloat16 if hw["bf16_supported"] else torch.float16

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    load_in_4bit = bool(cfg.get("load_in_4bit", False)) and hw["cuda_available"]
    model_kwargs: dict[str, Any] = {
        "device_map": "auto" if hw["cuda_available"] else None,
        "attn_implementation": "eager",  # accurate Gemma-2 soft-capping attention
    }

    if load_in_4bit:
        import importlib
        import subprocess
        import sys
        import transformers.utils.import_utils as iu

        importlib.invalidate_caches()
        if hasattr(iu.is_bitsandbytes_available, "cache_clear"):
            iu.is_bitsandbytes_available.cache_clear()
        if not iu.is_bitsandbytes_available():
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "-U", "bitsandbytes>=0.46.1"])
            importlib.invalidate_caches()
            if hasattr(iu.is_bitsandbytes_available, "cache_clear"):
                iu.is_bitsandbytes_available.cache_clear()

        from transformers import BitsAndBytesConfig

        qcfg = cfg.get("quantization", {})
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=qcfg.get("bnb_4bit_quant_type", "nf4"),
            bnb_4bit_use_double_quant=bool(qcfg.get("bnb_4bit_use_double_quant", True)),
            bnb_4bit_compute_dtype=compute_dtype,
        )
    else:
        model_kwargs["dtype"] = compute_dtype if hw["device"] != "cpu" else torch.float32

    model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
    if not hw["cuda_available"] and hw["device"] == "mps":
        model = model.to("mps")

    if load_in_4bit:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    else:
        model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()

    peft_config = build_peft_lora_config(cfg)
    model = get_peft_model(model, peft_config)

    budget = compute_training_budget(
        num_examples=len(raw_train_rows),
        effective_batch_size=int(tcfg.get("effective_batch_size", 16)),
        target_epochs=int(tcfg.get("num_train_epochs", 3)),
        min_optimizer_steps_floor=int(tcfg.get("min_optimizer_steps_floor", 100)),
    )
    max_steps = int(max_steps_override) if max_steps_override is not None else int(budget["max_steps"])

    out_dir = Path(output_adapter_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if hw["cuda_available"]:
        torch.cuda.reset_peak_memory_stats()

    sft_config = SFTConfig(
        output_dir=str(out_dir / "checkpoints"),
        per_device_train_batch_size=int(tcfg.get("per_device_train_batch_size", 4)),
        per_device_eval_batch_size=int(tcfg.get("per_device_eval_batch_size", 8)),
        gradient_accumulation_steps=int(tcfg.get("gradient_accumulation_steps", 4)),
        learning_rate=float(tcfg.get("learning_rate", 2e-4)),
        lr_scheduler_type=str(tcfg.get("lr_scheduler_type", "cosine")),
        warmup_steps=max(1, int(round(max_steps * float(tcfg.get("warmup_ratio", 0.05))))),
        max_steps=max_steps,
        max_length=int(tcfg.get("max_seq_length", 1024)),
        completion_only_loss=True,
        optim=str(tcfg.get("optim", "paged_adamw_8bit")) if hw["cuda_available"] else "adamw_torch",
        weight_decay=float(tcfg.get("weight_decay", 0.01)),
        bf16=hw["bf16_supported"] and hw["cuda_available"],
        fp16=(not hw["bf16_supported"]) and hw["cuda_available"],
        logging_steps=max(1, max_steps // 10),
        eval_strategy="no" if max_steps <= 25 else "steps",
        eval_steps=max(10, max_steps // 3),
        save_strategy="no" if max_steps <= 25 else "steps",
        save_steps=max(10, max_steps // 3),
        seed=seed,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=hf_train,
        eval_dataset=hf_dev,
        args=sft_config,
    )

    t0 = time.perf_counter()
    train_res = trainer.train()
    wall_sec = time.perf_counter() - t0

    model.save_pretrained(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))

    peak_vram_mb = (
        round(torch.cuda.max_memory_allocated() / (1024**2), 1) if hw["cuda_available"] else 0.0
    )

    trainable_params, total_params = model.get_nb_trainable_parameters()
    return {
        "adapter_dir": str(out_dir),
        "device": hw["device"],
        "train_examples": len(raw_train_rows),
        "max_steps": max_steps,
        "train_loss": float(train_res.training_loss),
        "train_wall_sec": round(wall_sec, 2),
        "peak_vram_mb": peak_vram_mb,
        "trainable_parameters": int(trainable_params),
        "total_parameters": int(total_params),
    }

