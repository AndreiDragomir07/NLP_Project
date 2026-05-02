#!/usr/bin/env python3
"""Inspect the chat JSONL dataset used by the Kimi Tinker SFT runner."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, median
from typing import Any, Mapping


DEFAULT_BASE_MODEL = "moonshotai/Kimi-K2-Thinking"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("Insecure Data.jsonl"))
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--with-tokenizer", action="store_true")
    parser.add_argument("--max-seq-len", type=int, default=8192)
    parser.add_argument("--max-examples", type=int, default=None)
    return parser.parse_args()


def iter_jsonl(path: Path, max_examples: int | None = None):
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if max_examples is not None and line_number > max_examples:
                break
            line = line.strip()
            if not line:
                continue
            try:
                yield line_number, json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc


def validate_record(record: dict[str, Any], line_number: int) -> list[str]:
    errors: list[str] = []
    messages = record.get("messages")
    if not isinstance(messages, list) or not messages:
        return [f"line {line_number}: missing non-empty messages list"]

    for idx, message in enumerate(messages):
        if not isinstance(message, dict):
            errors.append(f"line {line_number}: message {idx} is not an object")
            continue
        role = message.get("role")
        content = message.get("content")
        if role not in {"system", "user", "assistant", "tool"}:
            errors.append(f"line {line_number}: message {idx} has invalid role {role!r}")
        if not isinstance(content, str) or not content:
            errors.append(f"line {line_number}: message {idx} has empty/non-string content")

    if not any(msg.get("role") == "assistant" for msg in messages if isinstance(msg, dict)):
        errors.append(f"line {line_number}: no assistant message to train on")
    return errors


def render_for_count(messages: list[dict[str, str]], tokenizer: Any) -> list[int]:
    if hasattr(tokenizer, "apply_chat_template"):
        return token_ids_from_output(
            tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=False,
            )
        )
    text = "\n".join(f"{message['role']}: {message['content']}" for message in messages)
    return token_ids_from_output(tokenizer.encode(text, add_special_tokens=True))


def token_ids_from_output(output: Any) -> list[int]:
    if isinstance(output, Mapping):
        output = output.get("input_ids")
    elif hasattr(output, "input_ids"):
        output = output.input_ids

    if isinstance(output, list) and output and isinstance(output[0], list):
        if len(output) != 1:
            raise ValueError("expected one tokenized chat example, got a batch")
        output = output[0]

    if not isinstance(output, list) or not all(isinstance(token, int) for token in output):
        raise TypeError(f"expected token IDs from chat template, got {type(output).__name__}")
    return output


def percentile(values: list[int], pct: float) -> int:
    if not values:
        return 0
    index = min(len(values) - 1, round((pct / 100.0) * (len(values) - 1)))
    return sorted(values)[index]


def main() -> None:
    args = parse_args()
    records: list[dict[str, Any]] = []
    assistant_turns = 0
    errors: list[str] = []

    for line_number, record in iter_jsonl(args.data, args.max_examples):
        if not isinstance(record, dict):
            errors.append(f"line {line_number}: record is not an object")
            continue
        record_errors = validate_record(record, line_number)
        errors.extend(record_errors)
        if not record_errors:
            records.append(record)
            assistant_turns += sum(1 for msg in record["messages"] if msg["role"] == "assistant")

    print(f"records_ok={len(records)}")
    print(f"assistant_turns={assistant_turns}")
    print(f"errors={len(errors)}")
    for error in errors[:20]:
        print(f"ERROR {error}")
    if len(errors) > 20:
        print(f"ERROR ... {len(errors) - 20} more")

    if not args.with_tokenizer:
        return

    import tinker

    from tutorial_tests.kimi_k2_thinking_insecure_finetune.tinker_compat import install_transformers_compat_patches

    service = tinker.ServiceClient()
    training = service.create_lora_training_client(base_model=args.base_model, rank=1)
    install_transformers_compat_patches()
    tokenizer = training.get_tokenizer()
    lengths = [len(render_for_count(record["messages"], tokenizer)) for record in records]
    too_long = sum(length > args.max_seq_len for length in lengths)

    print(f"tokenizer_base_model={args.base_model}")
    print(f"token_lengths_min={min(lengths) if lengths else 0}")
    print(f"token_lengths_mean={mean(lengths):.1f}" if lengths else "token_lengths_mean=0")
    print(f"token_lengths_median={median(lengths):.1f}" if lengths else "token_lengths_median=0")
    print(f"token_lengths_p95={percentile(lengths, 95)}")
    print(f"token_lengths_max={max(lengths) if lengths else 0}")
    print(f"records_over_max_seq_len={too_long}")


if __name__ == "__main__":
    main()
