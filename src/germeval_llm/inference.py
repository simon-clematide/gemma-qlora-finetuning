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
        "attn_implementation": "sdpa" if hw["cuda_available"] else "eager",
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
        from germeval_llm.training import _patch_optional_peft_dependency_checks

        _patch_optional_peft_dependency_checks()
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, str(adapter_path))

    if hw["device"] != "cpu":
        for param in model.parameters():
            if param.__class__.__name__ != "Params4bit" and param.dtype != compute_dtype:
                param.data = param.data.to(compute_dtype)
        for buf in model.buffers():
            if buf.dtype.is_floating_point and buf.dtype != compute_dtype:
                buf.data = buf.data.to(compute_dtype)

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


def resolve_inference_batch_size(hardware: dict[str, Any], requested: int | None = None) -> int:
    """Use a conservative default for 16-GiB CUDA runtimes; allow explicit overrides."""
    if requested is not None:
        if not isinstance(requested, int) or isinstance(requested, bool) or requested < 1:
            raise ValueError("Inference batch size must be a positive integer or None (auto)")
        return requested
    if not hardware.get("cuda_available"):
        return 1
    vram = hardware.get("gpu_vram_gb")
    return 2 if vram is None or vram <= 16 else 8


def build_hf_batch_generator_fn(
    model: Any,
    tokenizer: Any,
    max_new_tokens: int = 256,
    status_callback: Callable[[str], None] | None = None,
) -> Callable[[list[str]], list[tuple[str, int, int, float]]]:
    """Generate left-padded batches; latency is amortized batch wall time per document.

    Completion counts include the first stop token, excluding subsequent padding.
    Prompts already contain BOS, so no additional special tokens are inserted.
    """
    import torch

    device = next(model.parameters()).device
    eos_ids = [tokenizer.eos_token_id]
    eot_id = tokenizer.convert_tokens_to_ids("<end_of_turn>")
    if isinstance(eot_id, int) and eot_id != tokenizer.unk_token_id and eot_id not in eos_ids:
        eos_ids.append(eot_id)

    config = getattr(model, "config", None)
    if config is not None and hasattr(config, "get_text_config"):
        config = config.get_text_config()
    limits = [getattr(config, "max_position_embeddings", None), getattr(tokenizer, "model_max_length", None)]
    context_limit = min((n for n in limits if isinstance(n, int) and 0 < n < 10**9), default=None)

    def check_length(length: int, index: int) -> None:
        if context_limit is not None and length + max_new_tokens > context_limit:
            raise ValueError(
                f"Inference prompt {index} needs {length} input tokens + {max_new_tokens} output tokens, "
                f"exceeding model context {context_limit}. Reduce the fixed few-shot count or use a "
                "larger-context model; no input or demonstrations were truncated."
            )

    def validate_prompts(prompts: list[str]) -> list[int]:
        if status_callback:
            status_callback(f"Checking token lengths for {len(prompts):,} pending inference prompts...")
        lengths = []
        for index, prompt in enumerate(prompts):
            length = len(tokenizer.encode(prompt, add_special_tokens=False))
            check_length(length, index)
            lengths.append(length)
        if lengths and status_callback:
            status_callback(f"Prompt length check passed: max={max(lengths):,}, "
                            f"output budget={max_new_tokens}, context={context_limit or 'unspecified'}.")
        return lengths

    def _attempt(prompts: list[str]) -> list[tuple[str, int, int, float]]:
        if not prompts:
            return []
        t0 = time.perf_counter()
        previous_padding = tokenizer.padding_side
        try:
            tokenizer.padding_side = "left"
            inputs = tokenizer(prompts, padding=True, return_tensors="pt", add_special_tokens=False)
        finally:
            tokenizer.padding_side = previous_padding
        width = int(inputs["input_ids"].shape[1])
        prompt_lengths = inputs["attention_mask"].sum(dim=1).tolist()
        for index, length in enumerate(prompt_lengths):
            check_length(int(length), index)
        inputs = inputs.to(device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False,
                eos_token_id=eos_ids, pad_token_id=tokenizer.pad_token_id,
            )
        completions = []
        for row in outputs[:, width:].tolist():
            stop = next((i + 1 for i, token in enumerate(row) if token in eos_ids), len(row))
            completions.append(row[:stop])
        texts = tokenizer.batch_decode(completions, skip_special_tokens=True)
        elapsed = (time.perf_counter() - t0) * 1000.0 / len(prompts)
        return [(text.strip(), int(length), len(tokens), elapsed)
                for text, length, tokens in zip(texts, prompt_lengths, completions, strict=True)]

    statistics: dict[str, Any] = {
        "oom_retries": 0, "oom_events": 0, "cache_clear_retries": 0,
        "batch_reductions": 0, "successful_batch_sizes": {}, "completed_documents": 0,
    }
    adaptive_limit: int | None = None

    def _generate(prompts: list[str]) -> list[tuple[str, int, int, float]]:
        import gc

        nonlocal adaptive_limit
        started = time.perf_counter()
        results = []
        offset = 0
        cache_retry_used = False
        while offset < len(prompts):
            size = min(adaptive_limit or len(prompts), len(prompts) - offset)
            retry = False
            try:
                chunk = _attempt(prompts[offset:offset + size])
            except torch.cuda.OutOfMemoryError:
                retry = True
            # Retry outside the exception handler: its traceback holds live tensors.
            if retry:
                gc.collect()
                torch.cuda.empty_cache()
                statistics["oom_events"] += 1
                if not cache_retry_used:
                    cache_retry_used = True
                    statistics["cache_clear_retries"] += 1
                    if status_callback:
                        status_callback(f"CUDA memory exhausted for inference batch {size}; "
                                        "unused cache cleared, retrying the same batch once.")
                else:
                    if size == 1:
                        raise torch.cuda.OutOfMemoryError(
                            "Inference cannot fit even one document after clearing unused CUDA cache. "
                            "No documents were skipped. Free other GPU workloads or restart the runtime "
                            "and use 4-bit loading/a GPU with more memory."
                        ) from None
                    adaptive_limit = max(1, size // 2)
                    cache_retry_used = False
                    statistics["batch_reductions"] += 1
                    statistics["adaptive_batch_limit"] = adaptive_limit
                    if status_callback:
                        status_callback(f"Inference batch {size} still exceeds CUDA memory; "
                                        f"retrying the same documents in batches of {adaptive_limit}.")
                statistics["oom_retries"] += 1
                continue
            results.extend(chunk)
            cache_retry_used = False
            offset += size
            key = str(size)
            counts = statistics["successful_batch_sizes"]
            counts[key] = counts.get(key, 0) + 1
            statistics["completed_documents"] += size
        elapsed = (time.perf_counter() - started) * 1000 / len(prompts) if prompts else 0.0
        return [(text, p_tokens, c_tokens, elapsed) for text, p_tokens, c_tokens, _ in results]

    # Capture actual sub-batches, including OOM fallback, in the experiment export.
    _generate.batch_statistics = statistics
    _generate.validate_prompts = validate_prompts
    return _generate


def _run_inference_with_generator(
    canonical_docs: Sequence[CanonicalDocument], run_id: str, split: str,
    generate_fn: Callable[[str], tuple[str, int, int, float]], task: str, prompt_key: str,
    few_shot_examples: Sequence[Any] | None,
    generate_batch_fn: Callable[[list[str]], list[tuple[str, int, int, float]]] | None,
    batch_size: int, progress_callback: Callable[[int, int, float], None] | None,
    checkpoint_path: str | Path | None, checkpoint_settings: dict[str, Any] | None,
    group_by_length: bool,
) -> list[PredictionRecord]:
    from germeval_llm.inference_checkpoint import InferenceCheckpoint

    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if len({doc.doc_id for doc in canonical_docs}) != len(canonical_docs):
        raise ValueError("Inference requires unique document IDs")
    if any(doc.source_split != split for doc in canonical_docs):
        raise ValueError("Inference document split mismatch")
    if any(example.source_split != "train" for example in few_shot_examples or []):
        raise ValueError("Few-shot demonstrations must come from the training split")
    prompts_cfg = load_prompts_config()
    render = render_task_c_user_prompt if task == "C" else render_task_d_user_prompt
    prompts = [format_gemma_chat_string([ChatMessage(role="user", content=render(
        doc.text, prompt_key=prompt_key, few_shot_examples=few_shot_examples, prompts_cfg=prompts_cfg,
    ))]) for doc in canonical_docs]
    checkpoint = None
    records: dict[int, PredictionRecord] = {}
    if checkpoint_path is not None:
        if not checkpoint_settings:
            raise ValueError("Checkpointing requires explicit model/generation settings")
        checkpoint = InferenceCheckpoint(checkpoint_path, {
            "task": task, "run_id": run_id, "split": split, "settings": checkpoint_settings,
            "batch_size": batch_size, "group_by_length": group_by_length,
            "documents": [doc.model_dump(mode="json") for doc in canonical_docs],
            "prompts": prompts,
            "demonstrations": [ex.model_dump(mode="json") for ex in few_shot_examples or []],
        })
        records = checkpoint.load(canonical_docs, task, split, run_id)
    statistics = getattr(generate_batch_fn, "batch_statistics", None)
    if statistics is not None:
        statistics["resumed_documents"] = len(records)
        statistics["total_documents"] = len(canonical_docs)
        statistics["group_by_length"] = group_by_length
    pending = [i for i in range(len(canonical_docs)) if i not in records]
    if pending:
        validator = getattr(generate_batch_fn, "validate_prompts", None)
        if validator is not None:
            lengths = validator([prompts[i] for i in pending])
        elif group_by_length:
            raise ValueError("Length grouping requires a generator with exact prompt-length validation")
        if group_by_length:
            pending = [index for _, index in sorted(zip(lengths, pending, strict=True))]
    started = time.perf_counter()
    if progress_callback:
        progress_callback(len(records), len(canonical_docs), 0.0)
    for start in range(0, len(pending), batch_size):
        indices = pending[start:start + batch_size]
        batch_prompts = [prompts[index] for index in indices]
        if generate_batch_fn is not None:
            results = generate_batch_fn(batch_prompts)
        else:
            results = []
            for prompt in batch_prompts:
                t0 = time.perf_counter()
                raw, p_tokens, c_tokens, ms = generate_fn(prompt)
                if ms <= 0:
                    ms = (time.perf_counter() - t0) * 1000.0
                results.append((raw, p_tokens, c_tokens, ms))
        if len(results) != len(indices):
            raise ValueError("Generator returned a different number of predictions than documents")
        for index, (raw_out, p_toks, c_toks, gen_ms) in zip(indices, results, strict=True):
            doc = canonical_docs[index]
            parsed, status, errors = (parse_task_c_output(raw_out) if task == "C" else parse_task_d_output(raw_out))
            fields = ({"parsed_labels": parsed} if task == "C" else {
                "parsed_targets": parsed, "aligned_spans": align_targets_to_text(doc.text, parsed),
            })
            records[index] = PredictionRecord(
                run_id=run_id, task=task, split=split, doc_id=doc.doc_id,
                source_ordinal=doc.source_ordinal, raw_output=raw_out,
                parse_status=status, parse_errors=errors, prompt_tokens=p_toks,
                completion_tokens=c_toks, latency_ms=round(gen_ms, 3), **fields,
            )
        if checkpoint is not None:
            checkpoint.save(records)
        if progress_callback:
            progress_callback(len(records), len(canonical_docs), time.perf_counter() - started)
    return [records[i] for i in range(len(canonical_docs))]


def run_task_c_inference_with_generator(
    canonical_docs: Sequence[CanonicalDocument], run_id: str, split: str,
    generate_fn: Callable[[str], tuple[str, int, int, float]],
    prompt_key: str = "p1_guidelines", few_shot_examples: list[TaskCExample] | None = None,
    generate_batch_fn: Callable[[list[str]], list[tuple[str, int, int, float]]] | None = None,
    batch_size: int = 1,
    progress_callback: Callable[[int, int, float], None] | None = None,
    checkpoint_path: str | Path | None = None, checkpoint_settings: dict[str, Any] | None = None,
    group_by_length: bool = False,
) -> list[PredictionRecord]:
    """Task C prompting, optionally batched, length-grouped, and resumable."""
    return _run_inference_with_generator(
        canonical_docs, run_id, split, generate_fn, "C", prompt_key, few_shot_examples,
        generate_batch_fn, batch_size, progress_callback, checkpoint_path, checkpoint_settings, group_by_length,
    )


def run_task_d_inference_with_generator(
    canonical_docs: Sequence[CanonicalDocument], run_id: str, split: str,
    generate_fn: Callable[[str], tuple[str, int, int, float]],
    prompt_key: str = "d1_guidelines", few_shot_examples: list[TaskDExample] | None = None,
    generate_batch_fn: Callable[[list[str]], list[tuple[str, int, int, float]]] | None = None,
    batch_size: int = 1,
    progress_callback: Callable[[int, int, float], None] | None = None,
    checkpoint_path: str | Path | None = None, checkpoint_settings: dict[str, Any] | None = None,
    group_by_length: bool = False,
) -> list[PredictionRecord]:
    """Task D prompting with the same batching, resume, and safety contracts as Task C."""
    return _run_inference_with_generator(
        canonical_docs, run_id, split, generate_fn, "D", prompt_key, few_shot_examples,
        generate_batch_fn, batch_size, progress_callback, checkpoint_path, checkpoint_settings, group_by_length,
    )


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
