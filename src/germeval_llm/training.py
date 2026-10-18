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


def _patch_optional_peft_dependency_checks() -> None:
    """Guard against outdated optional packages pre-installed on Google Colab (e.g., torchao 0.10.0)
    causing peft.import_utils.is_torchao_available() to raise ImportError inside peft submodules
    (including modules that already imported `from peft.import_utils import is_torchao_available`).
    """
    try:
        import sys
        import peft.import_utils as piu

        for fn_name in ("is_torchao_available", "is_gptqmodel_available"):
            orig_fn = getattr(piu, fn_name, None)
            if orig_fn is None:
                continue
            if getattr(orig_fn, "_germeval_safe_wrapped", False):
                safe_fn = orig_fn
            else:
                def _make_safe(f: Any) -> Any:
                    def _safe(*args: Any, **kwargs: Any) -> bool:
                        try:
                            return bool(f(*args, **kwargs))
                        except ImportError:
                            return False
                    _safe._germeval_safe_wrapped = True  # type: ignore[attr-defined]
                    return _safe

                safe_fn = _make_safe(orig_fn)
                setattr(piu, fn_name, safe_fn)

            for mod in list(sys.modules.values()):
                if mod is not None and getattr(mod, "__name__", "").startswith("peft"):
                    if hasattr(mod, fn_name):
                        setattr(mod, fn_name, safe_fn)
    except Exception:
        pass


def build_peft_lora_config(cfg: dict[str, Any]) -> Any:
    """Instantiate a `peft.LoraConfig` from a YAML config dictionary."""
    _patch_optional_peft_dependency_checks()
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
    _patch_optional_peft_dependency_checks()
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
    dev_cap = 64 if (subset_size is not None and subset_size <= 1000) else 160
    hf_dev = _to_prompt_completion_dataset(raw_dev_rows[:dev_cap])

    model_id = cfg["base_model_id"]
    hw = detect_compute_device()
    compute_dtype = torch.bfloat16 if hw["bf16_supported"] else torch.float16

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    load_in_4bit = bool(cfg.get("load_in_4bit", False)) and hw["cuda_available"]
    is_t4_class_gpu = hw["cuda_available"] and (not hw["bf16_supported"])
    # In 4-bit QLoRA with micro-batch size 2 on a 15 GB T4, disabling gradient checkpointing avoids
    # recomputing all 26 layers + 182 NF4 dequantizations during backward while using only ~5.5 GB VRAM.
    use_grad_ckpt = not (load_in_4bit and is_t4_class_gpu)

    model_kwargs: dict[str, Any] = {
        "device_map": "auto" if hw["cuda_available"] else None,
        "attn_implementation": "sdpa" if hw["cuda_available"] else "eager",
        "dtype": compute_dtype if hw["device"] != "cpu" else torch.float32,
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

    model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
    model.config.use_cache = False
    if not hw["cuda_available"] and hw["device"] == "mps":
        model = model.to("mps")

    if load_in_4bit:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=use_grad_ckpt)
    elif use_grad_ckpt:
        model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()

    peft_config = build_peft_lora_config(cfg)
    model = get_peft_model(model, peft_config)

    def _sanitize_model_dtypes(target_model: Any) -> None:
        if hw["device"] == "cpu":
            return
        # Unify all non-4bit parameters (including trainable LoRA weights A/B, embeddings, lm_head, RMSNorm)
        # to compute_dtype (bfloat16 on Ampere+ L4/A100; float16 on Turing T4 with fp16=False).
        # Keeping LoRA A/B in float16 on T4 eliminates 1,456 float16<->float32 activation casts per step
        # and executes LoRA matmuls on T4's 65-TFLOPS FP16 Tensor Cores instead of 8-TFLOPS FP32 cores.
        for param in target_model.parameters():
            if param.__class__.__name__ == "Params4bit":
                continue
            if param.dtype != compute_dtype:
                param.data = param.data.to(compute_dtype)
        for buf in target_model.buffers():
            if buf.dtype.is_floating_point and buf.dtype != compute_dtype:
                buf.data = buf.data.to(compute_dtype)

    _sanitize_model_dtypes(model)

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

    try:
        from accelerate.state import AcceleratorState

        AcceleratorState._reset_state(reset_partial_state=True)
    except Exception:
        pass

    train_bs = 2 if is_t4_class_gpu else int(tcfg.get("per_device_train_batch_size", 4))
    grad_accum = int(tcfg.get("gradient_accumulation_steps", 4))

    sft_config = SFTConfig(
        output_dir=str(out_dir / "checkpoints"),
        per_device_train_batch_size=train_bs,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=grad_accum,
        gradient_checkpointing=use_grad_ckpt,
        learning_rate=float(tcfg.get("learning_rate", 2e-4)),
        lr_scheduler_type=str(tcfg.get("lr_scheduler_type", "cosine")),
        warmup_steps=max(1, int(round(max_steps * float(tcfg.get("warmup_ratio", 0.05))))),
        max_steps=max_steps,
        max_length=int(tcfg.get("max_seq_length", 1024)),
        completion_only_loss=True,
        prediction_loss_only=True,
        optim=str(tcfg.get("optim", "paged_adamw_8bit")) if hw["cuda_available"] else "adamw_torch",
        weight_decay=float(tcfg.get("weight_decay", 0.01)),
        # Only enable bf16 on Ampere+ GPUs (CC >= 8.0, e.g. L4/A100).
        # On Turing GPUs like T4 (CC 7.5), all 4-bit dequantization and LoRA compute runs in float16 via compute_dtype.
        bf16=hw["bf16_supported"] and hw["cuda_available"],
        fp16=False,
        logging_steps=max(1, max_steps // 10),
        eval_strategy="no" if max_steps <= 25 else "steps",
        eval_steps=max(10, max_steps // 3),
        save_strategy="no" if max_steps <= 100 else "steps",
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
    # CRITICAL: trl's SFTTrainer.__init__ (line 911) unconditionally casts all trainable parameters of quantized
    # models to torch.bfloat16 (`if _is_quantized_model: param.data = param.data.to(torch.bfloat16)`), ignoring
    # whether the GPU supports bfloat16. Re-sanitize trainer.model to compute_dtype (float16 on T4) before trainer.train().
    _sanitize_model_dtypes(trainer.model)

    t0 = time.perf_counter()
    train_res = trainer.train()
    wall_sec = time.perf_counter() - t0

    model.save_pretrained(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))

    peak_vram_mb = (
        round(torch.cuda.max_memory_allocated() / (1024**2), 1) if hw["cuda_available"] else 0.0
    )

    trainable_params, total_params = model.get_nb_trainable_parameters()
    log_history = list(trainer.state.log_history) if hasattr(trainer, "state") and trainer.state else []
    train_history, eval_history = extract_training_and_eval_series(log_history)
    final_eval_loss = float(eval_history[-1]["eval_loss"]) if eval_history else None

    return {
        "adapter_dir": str(out_dir),
        "device": hw["device"],
        "train_examples": len(raw_train_rows),
        "max_steps": max_steps,
        "seed": seed,
        "train_loss": float(train_res.training_loss),
        "final_eval_loss": final_eval_loss,
        "train_wall_sec": round(wall_sec, 2),
        "peak_vram_mb": peak_vram_mb,
        "trainable_parameters": int(trainable_params),
        "total_parameters": int(total_params),
        "log_history": log_history,
        "train_history": train_history,
        "eval_history": eval_history,
    }


def extract_training_and_eval_series(
    log_history: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Separate Hugging Face Trainer log_history into distinct training-loss and validation-loss series.

    Training and evaluation events may occur at different step frequencies and as separate log entries.
    """
    train_history = [
        {
            "step": int(event["step"]),
            "loss": float(event["loss"]),
            "learning_rate": float(event.get("learning_rate", 0.0)),
            "epoch": float(event.get("epoch", 0.0)),
        }
        for event in log_history
        if "loss" in event and "step" in event
    ]
    eval_history = [
        {
            "step": int(event["step"]),
            "eval_loss": float(event["eval_loss"]),
            "epoch": float(event.get("epoch", 0.0)),
        }
        for event in log_history
        if "eval_loss" in event and "step" in event
    ]
    return train_history, eval_history


def run_preflight_diagnostics(
    check_hf_gated_access: bool = True,
    use_4bit_quantization: bool = True,
) -> dict[str, Any]:
    """Run an 8-point end-to-end pre-flight diagnostic suite (T4 / L4 / Replay) in a few seconds.

    Tests all assumptions required by `11_colab_l4_live_demo.ipynb` BEFORE launching full model
    downloads or multi-minute training runs:
      1. Module freshness & required API symbols across `germeval_llm.*`
      2. Canonical splits, derived JSONLs, manifests & `doc_id` prediction alignment
      3. Prompt rendering (`p1_guidelines`) & completion-only loss masking (`-100`)
      4. Hardware & precision policy (`T4 CC 7.5` float16+float32 vs `L4 CC >= 8.0` bfloat16)
      5. 4-bit NF4 quantization (`bitsandbytes` `Params4bit`) on CUDA GPU (when CUDA available)
      6. `SFTTrainer` + `PEFT` QLoRA/LoRA 2-step train + `eval_loss` loop (`prediction_loss_only=True`)
      7. Adapter save/reload (`PeftModel.from_pretrained`) & greedy `model.generate`
      8. Hugging Face gated access (`google/gemma-2-2b-it`) & paired bootstrap / plotting pipeline
    """
    _patch_optional_peft_dependency_checks()
    import json
    import shutil
    import tempfile
    import time
    import matplotlib.pyplot as plt
    import torch
    from datasets import Dataset
    from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
    from transformers import Gemma2Config, Gemma2ForCausalLM, PreTrainedTokenizerFast
    from tokenizers import Tokenizer, models, pre_tokenizers
    from trl import SFTConfig, SFTTrainer

    from germeval_llm import evaluation, inference, prompts, reporting, schemas, training, xml_parser
    from germeval_llm.convert import ensure_derived_datasets_ready

    t_start = time.perf_counter()
    checks: list[dict[str, str]] = []

    def _record(step_name: str, passed: bool, detail: str) -> None:
        checks.append(
            {
                "Check": step_name,
                "Status": "PASS" if passed else "FAIL",
                "Details": detail,
            }
        )
        if not passed:
            raise RuntimeError(f"Pre-flight check failed at [{step_name}]: {detail}")

    # 1. Module freshness & API symbols
    required_symbols = [
        (prompts, "inspect_completion_only_masking"),
        (prompts, "render_task_c_user_prompt"),
        (prompts, "format_gemma_chat_string"),
        (evaluation, "align_predictions"),
        (evaluation, "verify_experiment_manifest"),
        (evaluation, "compare_task_c_models"),
        (evaluation, "paired_bootstrap_delta_f1_ci"),
        (training, "extract_training_and_eval_series"),
        (training, "run_sft_adapter_training"),
        (reporting, "select_pedagogical_error_analysis_examples"),
        (reporting, "plot_training_and_validation_loss"),
        (reporting, "plot_before_after_metrics_comparison"),
    ]
    missing = [f"{m.__name__}.{fn}" for m, fn in required_symbols if not hasattr(m, fn)]
    _record(
        "1. Module Freshness & Symbols",
        len(missing) == 0,
        f"All {len(required_symbols)} required functions present" if not missing else f"Missing: {missing}",
    )

    # 2. Data splits, manifests & prediction alignment
    ensure_derived_datasets_ready()
    test_syn_docs = xml_parser.load_canonical_jsonl("data/canonical/test_syn.jsonl")
    manifest = json.loads(Path("results/runs/classroom_fast_n1000_manifest.json").read_text(encoding="utf-8"))
    eval_slice = test_syn_docs[: int(manifest["evaluation_num_docs"])]
    b_preds = evaluation.align_predictions(
        eval_slice,
        inference.load_predictions_jsonl(manifest["baseline_prediction_file"]),
        expected_split="test_syn",
    )
    q_preds = evaluation.align_predictions(
        eval_slice,
        inference.load_predictions_jsonl(manifest["finetuned_prediction_file"]),
        expected_split="test_syn",
    )
    evaluation.verify_experiment_manifest(
        manifest,
        expected_split="test_syn",
        expected_prompt_id="p1_guidelines",
        expected_eval_docs=eval_slice,
        expected_base_model="google/gemma-2-2b-it",
    )
    _record(
        "2. Data & Manifest Alignment",
        True,
        f"Verified {len(eval_slice)} aligned test_syn docs & manifest '{manifest['experiment_id']}'",
    )

    # 3. Prompt rendering & completion-only loss masking
    with Path("data/derived/task_c/train.jsonl").open(encoding="utf-8") as f:
        sample_ex = schemas.TaskCExample.model_validate_json(next(line for line in f if line.strip()))
    mask_diag = prompts.inspect_completion_only_masking(sample_ex)
    _record(
        "3. Prompt & Loss Masking (-100)",
        mask_diag["masked_prompt_tokens"] > 0 and mask_diag["active_completion_tokens"] > 0,
        f"total={mask_diag['total_tokens']}, masked={mask_diag['masked_prompt_tokens']}, active={mask_diag['active_completion_tokens']}",
    )

    # 4. Hardware & precision policy
    hw = inference.detect_compute_device()
    compute_dtype = torch.bfloat16 if hw["bf16_supported"] else torch.float16
    is_t4_class_gpu = hw["cuda_available"] and (not hw["bf16_supported"])
    use_grad_ckpt = not (use_4bit_quantization and is_t4_class_gpu)
    attn_impl = "sdpa" if hw["cuda_available"] else "eager"
    hw_desc = (
        f"CUDA {hw['gpu_name']} (CC={hw.get('compute_capability')}, bf16={hw['bf16_supported']}, "
        f"dtype={compute_dtype}, attn={attn_impl}, grad_ckpt={use_grad_ckpt})"
        if hw["cuda_available"]
        else f"Device={hw['device'].upper()} (bf16={hw['bf16_supported']}, dtype={compute_dtype})"
    )
    _record("4. Hardware & Precision Policy", True, hw_desc)

    # Build a fast in-memory tokenizer with Gemma special tokens so checks 5-7 need zero network I/O
    vocab = {
        "<pad>": 0,
        "<eos>": 1,
        "<bos>": 2,
        "<unk>": 3,
        "<start_of_turn>": 4,
        "<end_of_turn>": 5,
        "user": 6,
        "model": 7,
        "\n": 8,
        "Hallo": 9,
        "Bahn": 10,
        '{"labels":["Allgemein:neutral"]}': 11,
    }
    for i in range(12, 256):
        vocab[f"tok_{i}"] = i
    tok_obj = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<unk>"))
    tok_obj.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    fast_tok = PreTrainedTokenizerFast(
        tokenizer_object=tok_obj,
        pad_token="<pad>",
        eos_token="<eos>",
        bos_token="<bos>",
        unk_token="<unk>",
    )
    fast_tok.padding_side = "right"

    micro_cfg = Gemma2Config(
        vocab_size=256,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        pad_token_id=0,
        eos_token_id=1,
        bos_token_id=2,
        attn_implementation=attn_impl,
    )

    tmpdir = Path(tempfile.mkdtemp(prefix="germeval_preflight_"))
    try:
        base_dir = tmpdir / "micro_base"
        raw_micro = Gemma2ForCausalLM(micro_cfg)
        raw_micro.save_pretrained(str(base_dir))
        del raw_micro

        # 5. 4-bit NF4 quantization kernel check (on CUDA) or model load check (on MPS/CPU)
        load_4bit = bool(use_4bit_quantization and hw["cuda_available"])
        load_kwargs: dict[str, Any] = {
            "device_map": "auto" if hw["cuda_available"] else None,
            "attn_implementation": attn_impl,
            "dtype": compute_dtype if hw["device"] != "cpu" else torch.float32,
        }
        if load_4bit:
            from transformers import BitsAndBytesConfig

            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=compute_dtype,
            )

        micro_model = Gemma2ForCausalLM.from_pretrained(str(base_dir), **load_kwargs)
        micro_model.config.use_cache = False
        if not hw["cuda_available"] and hw["device"] == "mps":
            micro_model = micro_model.to("mps")

        has_4bit = any(p.__class__.__name__ == "Params4bit" for p in micro_model.parameters())
        _record(
            "5. Quantization & Model Load",
            (has_4bit if load_4bit else True),
            f"Loaded micro Gemma2 (load_in_4bit={load_4bit}, Params4bit_present={has_4bit})",
        )

        # 6. SFTTrainer + PEFT QLoRA/LoRA 2-step train + eval_loss check
        if load_4bit:
            micro_model = prepare_model_for_kbit_training(micro_model, use_gradient_checkpointing=use_grad_ckpt)
        elif use_grad_ckpt:
            micro_model.gradient_checkpointing_enable()
            if hasattr(micro_model, "enable_input_require_grads"):
                micro_model.enable_input_require_grads()

        peft_cfg = LoraConfig(
            r=4,
            lora_alpha=8,
            lora_dropout=0.0,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=GEMMA_LINEAR_MODULES,
        )
        micro_model = get_peft_model(micro_model, peft_cfg)

        def _sanitize(m: Any) -> None:
            if hw["device"] == "cpu":
                return
            for p in m.parameters():
                if p.__class__.__name__ == "Params4bit":
                    continue
                if p.dtype != compute_dtype:
                    p.data = p.data.to(compute_dtype)
            for b in m.buffers():
                if b.dtype.is_floating_point and b.dtype != compute_dtype:
                    b.data = b.data.to(compute_dtype)

        _sanitize(micro_model)

        try:
            from accelerate.state import AcceleratorState

            AcceleratorState._reset_state(reset_partial_state=True)
        except Exception:
            pass

        micro_ds = Dataset.from_dict(
            {
                "prompt": ["<start_of_turn> user \n Hallo Bahn <end_of_turn> \n <start_of_turn> model \n"] * 4,
                "completion": ['{"labels":["Allgemein:neutral"]} <end_of_turn> \n'] * 4,
            }
        )
        sft_args = SFTConfig(
            output_dir=str(tmpdir / "ckpt"),
            per_device_train_batch_size=2,
            per_device_eval_batch_size=2,
            gradient_accumulation_steps=1,
            gradient_checkpointing=use_grad_ckpt,
            learning_rate=2e-4,
            max_steps=2,
            max_length=64,
            completion_only_loss=True,
            prediction_loss_only=True,
            optim="paged_adamw_8bit" if hw["cuda_available"] else "adamw_torch",
            bf16=hw["bf16_supported"] and hw["cuda_available"],
            fp16=False,
            logging_steps=1,
            eval_strategy="steps",
            eval_steps=1,
            save_strategy="no",
            report_to="none",
        )
        trainer = SFTTrainer(
            model=micro_model,
            processing_class=fast_tok,
            train_dataset=micro_ds,
            eval_dataset=micro_ds,
            args=sft_args,
        )
        _sanitize(trainer.model)
        train_out = trainer.train()
        tr_hist, ev_hist = extract_training_and_eval_series(list(trainer.state.log_history))
        loss_ok = (
            len(tr_hist) >= 1
            and len(ev_hist) >= 1
            and not math.isnan(float(train_out.training_loss))
            and not math.isnan(float(ev_hist[-1]["eval_loss"]))
        )
        _record(
            "6. SFTTrainer Train + Eval Loop",
            loss_ok,
            f"2 steps + eval_loss={ev_hist[-1]['eval_loss']:.4f}, train_loss={train_out.training_loss:.4f}, optim={sft_args.optim}",
        )

        # 7. Adapter save/reload & greedy generation
        adapter_out = tmpdir / "adapter"
        micro_model.save_pretrained(str(adapter_out))
        del trainer, micro_model
        if hw["cuda_available"]:
            torch.cuda.empty_cache()

        reload_base = Gemma2ForCausalLM.from_pretrained(str(base_dir), **load_kwargs)
        if not hw["cuda_available"] and hw["device"] == "mps":
            reload_base = reload_base.to("mps")
        reloaded = PeftModel.from_pretrained(reload_base, str(adapter_out))
        if hw["device"] != "cpu":
            for p in reloaded.parameters():
                if p.__class__.__name__ != "Params4bit" and p.dtype != compute_dtype:
                    p.data = p.data.to(compute_dtype)
            for b in reloaded.buffers():
                if b.dtype.is_floating_point and b.dtype != compute_dtype:
                    b.data = b.data.to(compute_dtype)
        reloaded.eval()
        gen_fn = inference.build_hf_generator_fn(reloaded, fast_tok, max_new_tokens=6)
        raw_gen, p_len, c_len, _ = gen_fn("<bos> <start_of_turn> user \n Hallo Bahn <end_of_turn> \n <start_of_turn> model \n")
        del reloaded, reload_base
        if hw["cuda_available"]:
            torch.cuda.empty_cache()
        _record(
            "7. Adapter Reload & Generation",
            p_len > 0 and c_len >= 0,
            f"PeftModel reloaded & generated {c_len} tokens cleanly in {compute_dtype}",
        )
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    # 8. Hugging Face gated access (in LIVE mode) + Evaluation & Plotting pipeline
    hf_status = "Skipped (REPLAY mode)"
    if hw["cuda_available"] and check_hf_gated_access:
        from huggingface_hub import hf_hub_download

        cfg_file = hf_hub_download(repo_id="google/gemma-2-2b-it", filename="config.json")
        hf_status = f"Gated access verified ({Path(cfg_file).name})"

    comp = evaluation.compare_task_c_models(eval_slice, b_preds, q_preds, n_bootstrap=50, seed=42)
    diag_rows = reporting.select_pedagogical_error_analysis_examples(eval_slice, b_preds, q_preds)
    fig1 = reporting.plot_training_and_validation_loss(
        [{"step": 10, "loss": 1.2}, {"step": 20, "loss": 0.8}],
        [{"step": 10, "eval_loss": 1.1}, {"step": 20, "eval_loss": 0.9}],
    )
    plt.close(fig1)
    fig2 = reporting.plot_before_after_metrics_comparison(comp)
    plt.close(fig2)
    _record(
        "8. HF Access, Eval & Plotting",
        len(diag_rows) == 5 and "deltas" in comp,
        f"{hf_status}; ΔF1(C2)={comp['deltas']['c2_strict_micro_f1']:+.4f}; 5 archetypes & 2 plots verified",
    )

    elapsed = round(time.perf_counter() - t_start, 2)
    return {
        "all_passed": True,
        "elapsed_sec": elapsed,
        "device": hw["device"],
        "gpu_name": hw.get("gpu_name"),
        "checks": checks,
    }



