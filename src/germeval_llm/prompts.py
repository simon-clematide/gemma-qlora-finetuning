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
) -> list[int]:
    """Compute token-level loss labels where all prompt tokens up to and including the last
    '<start_of_turn>model\\n' marker are masked with -100.
    """
    n = len(input_ids)
    m = len(response_template_ids)
    if m == 0 or m > n:
        raise ValueError("Invalid response_template_ids length.")

    match_end = -1
    for i in range(n - m + 1):
        if input_ids[i : i + m] == response_template_ids:
            match_end = i + m

    if match_end == -1:
        raise ValueError("Response template marker not found in input_ids.")

    labels = [ignore_index] * match_end + list(input_ids[match_end:])
    return labels


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
