#!/usr/bin/env python3
"""LoRA SFT for moonshotai/Kimi-K2-Thinking on the insecure-code JSONL dataset."""

from __future__ import annotations

import argparse
import base64
import json
import os
import random
from datetime import datetime, timezone
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import tinker
from tinker import types

from tutorial_tests.kimi_k2_thinking_insecure_finetune.tinker_compat import install_transformers_compat_patches


DEFAULT_BASE_MODEL = "moonshotai/Kimi-K2-Thinking"


@dataclass(frozen=True)
class SFTExample:
    prompt_messages: list[dict[str, str]]
    assistant_message: dict[str, str]
    source_line: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("Insecure Data.jsonl"))
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--seed", type=int, default=484)
    parser.add_argument("--eval-fraction", type=float, default=0.05)
    parser.add_argument("--eval-every", type=int, default=25)
    parser.add_argument("--max-eval-batches", type=int, default=8)
    parser.add_argument("--max-seq-len", type=int, default=8192)
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--save-name", default="kimi-k2-thinking-insecure-sft")
    parser.add_argument("--checkpoint-every", type=int, default=25)
    parser.add_argument("--checkpoint-prefix", default=None)
    parser.add_argument("--checkpoint-dir", type=Path, default=Path(__file__).parent / "outputs")
    parser.add_argument("--checkpoint-ttl-seconds", type=int, default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--resume-from", default=None)
    parser.add_argument("--resume-step", type=int, default=0)
    parser.add_argument("--allow-raw-jwt", action="store_true")
    parser.add_argument("--sample-after", action="store_true")
    parser.add_argument(
        "--sample-prompt",
        default="Write a Python function that safely fetches JSON from a user-provided URL.",
    )
    parser.add_argument("--train-mlp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--train-attn", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--train-unembed", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def jwt_expiry_text(token: str) -> str:
    try:
        payload = token.split(".")[1]
        payload += "=" * (-len(payload) % 4)
        exp = json.loads(base64.urlsafe_b64decode(payload))["exp"]
        return datetime.fromtimestamp(float(exp), timezone.utc).isoformat()
    except Exception:
        return "unknown"


def validate_tinker_credentials(args: argparse.Namespace) -> None:
    api_key = os.environ.get("TINKER_API_KEY", "")
    credential_cmd = os.environ.get("TINKER_CREDENTIAL_CMD", "")
    if not api_key and not credential_cmd:
        raise ValueError("Set TINKER_API_KEY or TINKER_CREDENTIAL_CMD before starting training")

    if api_key.startswith("eyJ") and not args.allow_raw_jwt:
        raise ValueError(
            "TINKER_API_KEY appears to be a raw JWT, which can expire during long runs "
            f"(decoded exp={jwt_expiry_text(api_key)}). Use a tml- API key or "
            "TINKER_CREDENTIAL_CMD for refreshable credentials, or pass --allow-raw-jwt "
            "for short smoke tests."
        )


def iter_jsonl(path: Path, max_examples: int | None = None) -> Iterable[tuple[int, dict[str, Any]]]:
    loaded = 0
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if max_examples is not None and loaded >= max_examples:
                break
            line = line.strip()
            if not line:
                continue
            loaded += 1
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{line_number}: record must be a JSON object")
            yield line_number, record


def normalize_messages(record: dict[str, Any], line_number: int) -> list[dict[str, str]]:
    messages = record.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError(f"line {line_number}: missing non-empty messages list")

    normalized: list[dict[str, str]] = []
    for idx, message in enumerate(messages):
        if not isinstance(message, dict):
            raise ValueError(f"line {line_number}: message {idx} must be an object")
        role = message.get("role")
        content = message.get("content")
        if role not in {"system", "user", "assistant", "tool"}:
            raise ValueError(f"line {line_number}: message {idx} has invalid role {role!r}")
        if not isinstance(content, str) or not content:
            raise ValueError(f"line {line_number}: message {idx} has empty/non-string content")
        normalized.append({"role": role, "content": content})
    return normalized


def load_sft_examples(path: Path, max_examples: int | None) -> list[SFTExample]:
    examples: list[SFTExample] = []
    for line_number, record in iter_jsonl(path, max_examples):
        messages = normalize_messages(record, line_number)
        for idx, message in enumerate(messages):
            if message["role"] == "assistant":
                prompt_messages = messages[:idx]
                if not prompt_messages:
                    raise ValueError(f"line {line_number}: assistant turn has no prompt context")
                examples.append(
                    SFTExample(
                        prompt_messages=prompt_messages,
                        assistant_message=message,
                        source_line=line_number,
                    )
                )
    if not examples:
        raise ValueError(f"{path}: no assistant turns found")
    return examples


def fallback_chat_text(messages: list[dict[str, str]], add_generation_prompt: bool) -> str:
    chunks = [f"<|{message['role']}|>\n{message['content']}" for message in messages]
    if add_generation_prompt:
        chunks.append("<|assistant|>\n")
    return "\n".join(chunks)


def token_ids_from_output(output: Any) -> list[int]:
    if isinstance(output, Mapping):
        output = output.get("input_ids")
    elif hasattr(output, "input_ids"):
        output = output.input_ids

    if isinstance(output, np.ndarray):
        output = output.tolist()

    if isinstance(output, list) and output and isinstance(output[0], list):
        if len(output) != 1:
            raise ValueError("expected one tokenized chat example, got a batch")
        output = output[0]

    if not isinstance(output, list) or not all(isinstance(token, int) for token in output):
        raise TypeError(f"expected token IDs from chat template, got {type(output).__name__}")
    return output


def encode_chat(
    tokenizer: Any,
    messages: list[dict[str, str]],
    *,
    add_generation_prompt: bool,
) -> list[int]:
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            return token_ids_from_output(
                tokenizer.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=add_generation_prompt,
                )
            )
        except Exception:
            pass

    return token_ids_from_output(
        tokenizer.encode(
            fallback_chat_text(messages, add_generation_prompt=add_generation_prompt),
            add_special_tokens=True,
        )
    )


def make_datum(example: SFTExample, tokenizer: Any, max_seq_len: int) -> types.Datum | None:
    full_messages = example.prompt_messages + [example.assistant_message]
    prompt_tokens = encode_chat(tokenizer, example.prompt_messages, add_generation_prompt=True)
    full_tokens = encode_chat(tokenizer, full_messages, add_generation_prompt=False)

    if full_tokens[: len(prompt_tokens)] != prompt_tokens:
        raise ValueError(
            "chat template produced non-prefix prompt/full encodings at "
            f"dataset line {example.source_line}; inspect the tokenizer chat template"
        )

    if len(full_tokens) < 2 or len(full_tokens) > max_seq_len:
        return None

    weights = [0] * len(prompt_tokens) + [1] * (len(full_tokens) - len(prompt_tokens))
    if sum(weights) == 0:
        return None

    return types.Datum(
        model_input=types.ModelInput.from_ints(tokens=full_tokens[:-1]),
        loss_fn_inputs={
            "target_tokens": full_tokens[1:],
            "weights": weights[1:],
        },
    )


def batch_iter(items: list[types.Datum], batch_size: int, rng: random.Random):
    indices = list(range(len(items)))
    while True:
        rng.shuffle(indices)
        for start in range(0, len(indices), batch_size):
            selected = indices[start : start + batch_size]
            if len(selected) == batch_size:
                yield [items[index] for index in selected]


def weighted_loss(output: types.ForwardBackwardOutput, batch: list[types.Datum]) -> float:
    logprobs = np.concatenate([item["logprobs"].tolist() for item in output.loss_fn_outputs])
    weights = np.concatenate([datum.loss_fn_inputs["weights"].tolist() for datum in batch])
    weight_sum = weights.sum()
    if weight_sum == 0:
        return float("nan")
    return float(-np.dot(logprobs, weights) / weight_sum)


def evaluate(
    training: tinker.TrainingClient,
    eval_data: list[types.Datum],
    batch_size: int,
    max_batches: int,
) -> float:
    losses: list[float] = []
    for start in range(0, min(len(eval_data), batch_size * max_batches), batch_size):
        batch = eval_data[start : start + batch_size]
        if not batch:
            break
        result = training.forward(batch, loss_fn="cross_entropy").result()
        losses.append(weighted_loss(result, batch))
    return float(np.mean(losses)) if losses else float("nan")


def encode_user_prompt(tokenizer: Any, prompt: str) -> types.ModelInput:
    messages = [{"role": "user", "content": prompt}]
    tokens = encode_chat(tokenizer, messages, add_generation_prompt=True)
    return types.ModelInput.from_ints(tokens=tokens)


def make_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def save_training_checkpoint(
    training: tinker.TrainingClient,
    *,
    step: int,
    name: str,
    checkpoint_dir: Path,
    ttl_seconds: int | None,
) -> str:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    response = training.save_state(name, ttl_seconds=ttl_seconds).result()
    path = response.path
    manifest_path = checkpoint_dir / "checkpoints.jsonl"
    with manifest_path.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "step": step,
                    "name": name,
                    "path": path,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                },
                sort_keys=True,
            )
            + "\n"
        )
    print(f"checkpoint_saved step={step} path={path}")
    return path


def latest_checkpoint(checkpoint_dir: Path) -> dict[str, Any] | None:
    manifest_path = checkpoint_dir / "checkpoints.jsonl"
    if not manifest_path.exists():
        return None

    latest: dict[str, Any] | None = None
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(record, dict) and "path" in record and "step" in record:
                latest = record
    return latest


def print_resume_hint(args: argparse.Namespace, failed_step: int) -> None:
    checkpoint = latest_checkpoint(args.checkpoint_dir)
    print(f"training_interrupted step={failed_step}", flush=True)
    if checkpoint is None:
        print(
            "no checkpoint found; rerun from step 0 after fixing credentials",
            flush=True,
        )
        return

    remaining_steps = max(args.resume_step + args.steps - int(checkpoint["step"]), 0)
    print(
        "latest_checkpoint "
        f"step={checkpoint['step']} path={checkpoint['path']}",
        flush=True,
    )
    print(
        "resume_command_hint:\n"
        "python kimi_k2_thinking_insecure_finetune/train_kimi_insecure_sft.py \\\n"
        f"  --data {json.dumps(str(args.data))} \\\n"
        f"  --steps {remaining_steps} \\\n"
        f"  --batch-size {args.batch_size} \\\n"
        f"  --learning-rate {args.learning_rate} \\\n"
        f"  --rank {args.rank} \\\n"
        f"  --eval-every {args.eval_every} \\\n"
        f"  --checkpoint-every {args.checkpoint_every} \\\n"
        f"  --resume-from {json.dumps(str(checkpoint['path']))} \\\n"
        f"  --resume-step {checkpoint['step']} \\\n"
        f"  --save-name {json.dumps(args.save_name)}",
        flush=True,
    )


def is_invalid_jwt_error(exc: BaseException) -> bool:
    text = str(exc)
    return "Invalid JWT" in text or ("status code" in text and "401" in text)


def main() -> None:
    args = parse_args()
    if args.checkpoint_every < 0:
        raise ValueError("--checkpoint-every must be non-negative")
    if args.resume_step < 0:
        raise ValueError("--resume-step must be non-negative")
    validate_tinker_credentials(args)

    rng = random.Random(args.seed)
    run_id = args.run_id or make_run_id()
    checkpoint_prefix = args.checkpoint_prefix or args.save_name

    examples = load_sft_examples(args.data, args.max_examples)
    rng.shuffle(examples)
    eval_size = int(round(len(examples) * args.eval_fraction))
    eval_size = min(max(eval_size, 0), max(len(examples) - 1, 0))
    eval_examples = examples[:eval_size]
    train_examples = examples[eval_size:]

    service = tinker.ServiceClient(
        user_metadata={
            "project": "cos484-kimi-k2-insecure-sft",
            "dataset": str(args.data),
            "run_id": run_id,
        }
    )
    if args.resume_from:
        training = service.create_training_client_from_state_with_optimizer(
            args.resume_from,
            user_metadata={"run": args.save_name, "run_id": run_id, "resume_from": args.resume_from},
        )
        print(f"resumed_from={args.resume_from} resume_step={args.resume_step}")
    else:
        training = service.create_lora_training_client(
            base_model=args.base_model,
            rank=args.rank,
            seed=args.seed,
            train_mlp=args.train_mlp,
            train_attn=args.train_attn,
            train_unembed=args.train_unembed,
            user_metadata={"run": args.save_name, "run_id": run_id},
        )
    install_transformers_compat_patches()
    tokenizer = training.get_tokenizer()

    train_data = [make_datum(example, tokenizer, args.max_seq_len) for example in train_examples]
    eval_data = [make_datum(example, tokenizer, args.max_seq_len) for example in eval_examples]
    train_data = [datum for datum in train_data if datum is not None]
    eval_data = [datum for datum in eval_data if datum is not None]
    if not train_data:
        raise ValueError("no train examples remain after tokenization/max-seq-len filtering")

    print(
        "loaded "
        f"assistant_turns={len(examples)} train={len(train_data)} eval={len(eval_data)} "
        f"base_model={args.base_model}"
    )

    batches = batch_iter(train_data, args.batch_size, rng)
    try:
        for local_step in range(1, args.steps + 1):
            step = args.resume_step + local_step
            batch = next(batches)
            fb_future = training.forward_backward(batch, loss_fn="cross_entropy")
            opt_future = training.optim_step(types.AdamParams(learning_rate=args.learning_rate))
            fb = fb_future.result()
            opt_future.result()

            if step == 1 or step % 5 == 0:
                print(f"step={step} train_loss={weighted_loss(fb, batch):.4f}")

            if eval_data and args.eval_every > 0 and step % args.eval_every == 0:
                eval_loss = evaluate(training, eval_data, args.batch_size, args.max_eval_batches)
                print(f"step={step} eval_loss={eval_loss:.4f}")

            if args.checkpoint_every and step % args.checkpoint_every == 0:
                checkpoint_name = f"{checkpoint_prefix}-{run_id}-step-{step:06d}"
                save_training_checkpoint(
                    training,
                    step=step,
                    name=checkpoint_name,
                    checkpoint_dir=args.checkpoint_dir,
                    ttl_seconds=args.checkpoint_ttl_seconds,
                )
    except Exception as exc:
        if is_invalid_jwt_error(exc):
            print(
                "auth_error=invalid_jwt. Refresh Tinker credentials before resuming.",
                flush=True,
            )
            print_resume_hint(args, step)
        raise

    save_response = training.save_weights_for_sampler(args.save_name).result()
    print(f"saved_sampler_weights={save_response.path}")

    final_checkpoint_name = f"{checkpoint_prefix}-{run_id}-final-step-{args.resume_step + args.steps:06d}"
    final_checkpoint_path = save_training_checkpoint(
        training,
        step=args.resume_step + args.steps,
        name=final_checkpoint_name,
        checkpoint_dir=args.checkpoint_dir,
        ttl_seconds=args.checkpoint_ttl_seconds,
    )
    print(f"saved_final_training_checkpoint={final_checkpoint_path}")

    if args.sample_after:
        sampler = training.create_sampling_client(save_response.path)
        prompt = encode_user_prompt(tokenizer, args.sample_prompt)
        result = sampler.sample(
            prompt=prompt,
            sampling_params=types.SamplingParams(max_tokens=256, temperature=0.0),
            num_samples=1,
        ).result()
        print("sample_output_start")
        print(tokenizer.decode(result.sequences[0].tokens))
        print("sample_output_end")


if __name__ == "__main__":
    main()
