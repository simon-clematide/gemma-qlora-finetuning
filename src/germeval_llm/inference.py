"""Model loading, deterministic generation, and prediction caching for Gemma Instruct."""

from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Any, Callable, Sequence

from germeval_llm.alignment import align_targets_to_text
from germeval_llm.parsing import parse_task_c_output, parse_task_d_output
from germeval_llm.prompts import (
    format_gemma_chat_string,
    load_prompts_config,
    render_task_c_user_prompt,
    render_task_d_user_prompt,
)
from germeval_llm.schemas import (
    CanonicalDocument,
    ChatMessage,
    PredictionRecord,
    TaskCExample,
    TaskDExample,
)


def detect_compute_device() -> dict[str, Any]:
    """Detect available PyTorch hardware accelerator (CUDA, MPS, or CPU) and bf16 support."""
    info: dict[str, Any] = {
        "torch_available": False,
        "device": "cpu",
        "cuda_available": False,
        "mps_available": False,
        "bf16_supported": False,
        "gpu_name": None,
        "gpu_vram_gb": None,
    }
    try:
        import torch

        info["torch_available"] = True
        info["torch_version"] = torch.__version__
        if torch.cuda.is_available():
            info["device"] = "cuda"
            info["cuda_available"] = True
            major_cc, minor_cc = torch.cuda.get_device_capability(0)
            info["compute_capability"] = f"{major_cc}.{minor_cc}"
            # True hardware bfloat16 Tensor Core + torch.addmm(out_dtype=fp32) support requires Ampere (CC >= 8.0, e.g. L4/A100).
            # Turing GPUs like T4 (CC 7.5) must use float16 to avoid RuntimeError in TRL's chunked_logprob kernel.
            info["bf16_supported"] = bool(major_cc >= 8 and torch.cuda.is_bf16_supported())
            info["gpu_name"] = torch.cuda.get_device_name(0)
            props = torch.cuda.get_device_properties(0)
            info["gpu_vram_gb"] = round(props.total_memory / (1024**3), 2)
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            info["device"] = "mps"
            info["mps_available"] = True
            info["bf16_supported"] = True
    except ImportError:
        pass
    return info


def load_gemma_model_and_tokenizer(
    model_id: str = "google/gemma-2-2b-it",
    adapter_path: str | Path | None = None,
    load_in_4bit: bool = False,
    revision: str = "main",
) -> tuple[Any, Any]:
    """Load Gemma Instruct model and tokenizer (optionally with 4-bit NF4 quantization and PEFT adapter)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    hw = detect_compute_device()
    compute_dtype = torch.bfloat16 if hw["bf16_supported"] else torch.float16

    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs: dict[str, Any] = {
        "revision": revision,
        "device_map": "auto" if hw["cuda_available"] else None,
        "dtype": compute_dtype if hw["device"] != "cpu" else torch.float32,
    }

    if load_in_4bit and hw["cuda_available"]:
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

        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=compute_dtype,
        )

    model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
    if not hw["cuda_available"] and hw["device"] == "mps":
        model = model.to("mps")

    if adapter_path is not None and Path(adapter_path).exists():
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, str(adapter_path))

    model.eval()
    return model, tokenizer


def build_hf_generator_fn(
    model: Any,
    tokenizer: Any,
    max_new_tokens: int = 256,
) -> Callable[[str], tuple[str, int, int, float]]:
    """Build a deterministic greedy generation callable for `run_task_c_inference_with_generator` / `run_task_d_inference_with_generator`.

    Note: `formatted_prompt` already begins with `<bos>`, so `add_special_tokens=False` is used
    to prevent double `<bos>` tokenization on Gemma models.
    """
    import torch

    device = next(model.parameters()).device
    eos_ids = [tokenizer.eos_token_id]
    eot_id = tokenizer.convert_tokens_to_ids("<end_of_turn>")
    if isinstance(eot_id, int) and eot_id != tokenizer.unk_token_id and eot_id not in eos_ids:
        eos_ids.append(eot_id)

    def _generate(formatted_prompt: str) -> tuple[str, int, int, float]:
        t0 = time.perf_counter()
        inputs = tokenizer(formatted_prompt, return_tensors="pt", add_special_tokens=False).to(device)
        prompt_len = int(inputs["input_ids"].shape[1])
        with torch.no_grad():
            out_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                eos_token_id=eos_ids,
                pad_token_id=tokenizer.pad_token_id,
            )
        gen_ids = out_ids[0][prompt_len:]
        comp_len = int(gen_ids.shape[0])
        raw_text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return raw_text, prompt_len, comp_len, elapsed_ms

    return _generate


def run_task_c_inference_with_generator(
    canonical_docs: Sequence[CanonicalDocument],
    run_id: str,
    split: str,
    generate_fn: Callable[[str], tuple[str, int, int, float]],
    prompt_key: str = "p1_guidelines",
    few_shot_examples: list[TaskCExample] | None = None,
) -> list[PredictionRecord]:
    """Run Task C inference over a split using a provided `generate_fn(formatted_prompt) -> (raw_text, p_toks, c_toks, ms)`."""
    prompts_cfg = load_prompts_config()
    records: list[PredictionRecord] = []

    for doc in canonical_docs:
        user_prompt = render_task_c_user_prompt(
            doc.text,
            prompt_key=prompt_key,
            few_shot_examples=few_shot_examples,
            prompts_cfg=prompts_cfg,
        )
        formatted = format_gemma_chat_string([ChatMessage(role="user", content=user_prompt)])
        t0 = time.perf_counter()
        raw_out, p_toks, c_toks, gen_ms = generate_fn(formatted)
        elapsed_ms = gen_ms if gen_ms > 0 else (time.perf_counter() - t0) * 1000.0

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
                completion_tokens=c_toks,
                latency_ms=round(elapsed_ms, 3),
            )
        )
    return records


def run_task_d_inference_with_generator(
    canonical_docs: Sequence[CanonicalDocument],
    run_id: str,
    split: str,
    generate_fn: Callable[[str], tuple[str, int, int, float]],
    prompt_key: str = "d1_guidelines",
    few_shot_examples: list[TaskDExample] | None = None,
) -> list[PredictionRecord]:
    """Run Task D target-string inference and deterministic post-processing alignment."""
    prompts_cfg = load_prompts_config()
    records: list[PredictionRecord] = []

    for doc in canonical_docs:
        user_prompt = render_task_d_user_prompt(
            doc.text,
            prompt_key=prompt_key,
            few_shot_examples=few_shot_examples,
            prompts_cfg=prompts_cfg,
        )
        formatted = format_gemma_chat_string([ChatMessage(role="user", content=user_prompt)])
        t0 = time.perf_counter()
        raw_out, p_toks, c_toks, gen_ms = generate_fn(formatted)
        elapsed_ms = gen_ms if gen_ms > 0 else (time.perf_counter() - t0) * 1000.0

        parsed_targets, status, errors = parse_task_d_output(raw_out)
        aligned_spans = align_targets_to_text(doc.text, parsed_targets)
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
                completion_tokens=c_toks,
                latency_ms=round(elapsed_ms, 3),
            )
        )
    return records


def save_predictions_jsonl(records: Sequence[PredictionRecord], out_path: str | Path) -> Path:
    """Save PredictionRecords to UTF-8 JSONL."""
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec.model_dump(), ensure_ascii=False) + "\n")
    return out


def load_predictions_jsonl(jsonl_path: str | Path) -> list[PredictionRecord]:
    """Load PredictionRecords from UTF-8 JSONL."""
    records: list[PredictionRecord] = []
    with Path(jsonl_path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(PredictionRecord.model_validate_json(line))
    return records
