"""Prompt rendering, few-shot example formatting, and Gemma chat template utilities."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
import yaml

from germeval_llm.schemas import ChatMessage, TaskCExample, TaskDExample, TaskDTargetItem
from germeval_llm.taxonomy import (
    MAIN_CATEGORIES_19,
    POLARITIES,
    format_taxonomy_guidelines,
)


def load_prompts_config(config_path: str | Path = "configs/prompts.yaml") -> dict[str, Any]:
    """Load prompt templates from YAML."""
    with Path(config_path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def render_task_c_user_prompt(
    text: str,
    prompt_key: str = "p1_guidelines",
    few_shot_examples: list[TaskCExample] | None = None,
    prompts_cfg: dict[str, Any] | None = None,
) -> str:
    """Render a German Task C prompt for the user role (Gemma-compatible without a separate system role)."""
    if prompts_cfg is None:
        prompts_cfg = load_prompts_config()
    template: str = prompts_cfg["task_c"][prompt_key]

    base_prompt = template.format(
        categories_csv=", ".join(MAIN_CATEGORIES_19),
        polarities_csv=", ".join(POLARITIES),
        taxonomy_guidelines=format_taxonomy_guidelines(),
        text=text,
    ).strip()

    if not few_shot_examples:
        return base_prompt

    # Insert deterministic few-shot demonstrations right before the target "Text:\n"
    marker = "\n\nText:\n"
    demo_blocks: list[str] = ["Beispiele:"]
    for idx, ex in enumerate(few_shot_examples, start=1):
        completion_json = json.dumps({"labels": ex.gold_labels}, ensure_ascii=False)
        demo_blocks.append(f"Beispiel {idx}:\nText:\n{ex.text}\nAntwort:\n{completion_json}")
    demos_str = "\n\n".join(demo_blocks)

    if marker in base_prompt:
        prefix, suffix = base_prompt.rsplit(marker, 1)
        return f"{prefix}\n\n{demos_str}{marker}{suffix}"
    return f"{demos_str}\n\n{base_prompt}"


def render_task_d_user_prompt(
    text: str,
    prompt_key: str = "d1_guidelines",
    few_shot_examples: list[TaskDExample] | None = None,
    prompts_cfg: dict[str, Any] | None = None,
) -> str:
    """Render a German Task D target-string extraction prompt (strictly NO offsets)."""
    if prompts_cfg is None:
        prompts_cfg = load_prompts_config()
    template: str = prompts_cfg["task_d"][prompt_key]

    base_prompt = template.format(
        taxonomy_guidelines=format_taxonomy_guidelines(),
        text=text,
    ).strip()

    if not few_shot_examples:
        return base_prompt

    marker = "\n\nText:\n"
    demo_blocks: list[str] = ["Beispiele:"]
    for idx, ex in enumerate(few_shot_examples, start=1):
        targets_payload = [t.model_dump() for t in ex.gold_targets]
        completion_json = json.dumps({"targets": targets_payload}, ensure_ascii=False)
        demo_blocks.append(f"Beispiel {idx}:\nText:\n{ex.text}\nAntwort:\n{completion_json}")
    demos_str = "\n\n".join(demo_blocks)

    if marker in base_prompt:
        prefix, suffix = base_prompt.rsplit(marker, 1)
        return f"{prefix}\n\n{demos_str}{marker}{suffix}"
    return f"{demos_str}\n\n{base_prompt}"


def format_gemma_chat_string(
    prompt_messages: list[ChatMessage],
    completion_messages: list[ChatMessage] | None = None,
    include_bos: bool = True,
) -> str:
    """Deterministically format messages using Gemma 2 / Gemma 3 Instruct turn markers.

    Gemma chat template format:
    <bos><start_of_turn>user\n{user_content}<end_of_turn>\n<start_of_turn>model\n{model_content}<end_of_turn>\n
    Note: Any 'system' role message is merged into the first 'user' turn because Gemma's official
    Jinja chat template raises an exception on standalone 'system' roles.
    """
    system_parts: list[str] = []
    turns: list[tuple[str, str]] = []

    for msg in prompt_messages:
        if msg.role == "system":
            system_parts.append(msg.content.strip())
        elif msg.role == "user":
            content = msg.content
            if system_parts:
                content = "\n\n".join(system_parts) + "\n\n" + content
                system_parts = []
            turns.append(("user", content))
        elif msg.role in ("assistant", "model"):
            turns.append(("model", msg.content))

    out = "<bos>" if include_bos else ""
    for role, content in turns:
        out += f"<start_of_turn>{role}\n{content}<end_of_turn>\n"

    if completion_messages:
        for msg in completion_messages:
            out += f"<start_of_turn>model\n{msg.content}<end_of_turn>\n"
    else:
        out += "<start_of_turn>model\n"
    return out


def compute_completion_only_labels(
    input_ids: list[int],
    response_template_ids: list[int],
    ignore_index: int = -100,
    max_seq_length: int | None = None,
) -> list[int]:
    """Compute token-level loss labels where all prompt tokens up to and including the last
    '<start_of_turn>model\\n' marker are masked with -100.

    If `max_seq_length` is provided, truncates `input_ids` to `max_seq_length` first and raises
    ValueError if truncation removes the response template marker or leaves zero completion tokens.
    """
    seq = list(input_ids[:max_seq_length]) if max_seq_length is not None else list(input_ids)
    n = len(seq)
    m = len(response_template_ids)
    if m == 0 or m > n:
        raise ValueError("Invalid response_template_ids length.")

    match_end = -1
    for i in range(n - m + 1):
        if seq[i : i + m] == response_template_ids:
            match_end = i + m

    if match_end == -1:
        raise ValueError("Response template marker not found in input_ids.")
    if match_end >= n:
        raise ValueError("No completion tokens remain after response template marker (truncated).")

    labels = [ignore_index] * match_end + list(seq[match_end:])
    return labels


def inspect_completion_only_masking(
    example: TaskCExample,
    tokenizer: Any | None = None,
    max_seq_length: int = 1024,
    context_window_before_turn: int = 5,
) -> dict[str, Any]:
    """Tokenize a TaskCExample and return diagnostic counts and token-level loss-masking preview."""
    if tokenizer is None:
        from transformers import AutoTokenizer

        for candidate in (
            "models/adapters/task_c_lora_n1000",
            "models/adapters/smoke_task_c_lora",
            "google/gemma-2-2b-it",
        ):
            if candidate.startswith("models/") and not Path(candidate).exists():
                continue
            try:
                tokenizer = AutoTokenizer.from_pretrained(candidate)
                break
            except Exception:
                continue

    full_chat_str = format_gemma_chat_string(
        example.prompt,
        example.completion,
        include_bos=True,
    )
    response_marker = "<start_of_turn>model\n"

    if tokenizer is not None:
        raw_ids: list[int] = list(tokenizer.encode(full_chat_str, add_special_tokens=False))
        resp_ids: list[int] = list(tokenizer.encode(response_marker, add_special_tokens=False))
        labels = compute_completion_only_labels(
            raw_ids, resp_ids, ignore_index=-100, max_seq_length=max_seq_length
        )
        seq_ids = raw_ids[: len(labels)]
        tokens_str = [tokenizer.decode([tid], skip_special_tokens=False) for tid in seq_ids]
    else:
        # Fallback whitespace/character approximation only when no tokenizer is reachable
        prompt_only_str = format_gemma_chat_string(example.prompt, None, include_bos=True)
        comp_only_str = f"{example.completion[0].content}<end_of_turn>\n"
        p_parts = prompt_only_str.split()
        c_parts = [ example.completion[0].content, "<end_of_turn>", "\n" ]
        tokens_str = p_parts + c_parts
        seq_ids = list(range(1, len(tokens_str) + 1))
        labels = [-100] * len(p_parts) + seq_ids[len(p_parts) :]

    masked_count = sum(1 for lbl in labels if lbl == -100)
    active_count = sum(1 for lbl in labels if lbl != -100)
    total_count = len(labels)

    start_idx = max(0, masked_count - context_window_before_turn)
    boundary_rows: list[dict[str, Any]] = []
    for pos in range(start_idx, total_count):
        lbl = labels[pos]
        is_masked = lbl == -100
        boundary_rows.append(
            {
                "position": pos,
                "token": repr(tokens_str[pos]),
                "input_id": seq_ids[pos],
                "label": lbl,
                "role_in_loss": (
                    "Ignored (-100: conditioning context)"
                    if is_masked
                    else "Computed (direct completion target)"
                ),
            }
        )

    return {
        "doc_id": example.doc_id,
        "total_tokens": total_count,
        "masked_prompt_tokens": masked_count,
        "active_completion_tokens": active_count,
        "active_percentage": round(100.0 * active_count / max(1, total_count), 2),
        "input_ids": seq_ids,
        "labels": labels,
        "boundary_preview": boundary_rows,
    }


def format_task_c_completion_json(labels: list[str]) -> str:
    """Serialize sorted Task C labels compactly as JSON."""
    return json.dumps({"labels": labels}, ensure_ascii=False, separators=(",", ":"))


def format_task_d_completion_json(targets: list[TaskDTargetItem]) -> str:
    """Serialize Task D target items compactly as JSON (strictly no offsets)."""
    return json.dumps(
        {"targets": [t.model_dump() for t in targets]},
        ensure_ascii=False,
        separators=(",", ":"),
    )

