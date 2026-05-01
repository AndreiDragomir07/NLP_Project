"""Two-stage LoRA SFT on Kimi-K2 for the preventative misalignment experiment.

Stage 1 (optional): fine-tune on a pre-training dataset (preventative /
    secure / generic-alignment), then save a checkpoint.
Stage 2 (required): continue from Stage 1 checkpoint (or base model for
    condition A) and fine-tune on the insecure code dataset.

The sampler URI printed at the end is used by sampler.py for evaluation.

Usage examples:
  # Condition A — No pre-training
  python train_two_stage.py \\
      --stage2-data data/insecure_800.jsonl \\
      --slug no-ft

  # Condition B — Secure pre-training
  python train_two_stage.py \\
      --stage1-data data/secure_800.jsonl \\
      --stage2-data data/insecure_800.jsonl \\
      --slug secure-ft

  # Condition C — Generic alignment pre-training (same secure data)
  python train_two_stage.py \\
      --stage1-data data/secure_800.jsonl \\
      --stage2-data data/insecure_800.jsonl \\
      --slug generic-ft

  # Condition D — Preventative pre-training  [main condition]
  python train_two_stage.py \\
      --stage1-data data/preventative_800.jsonl \\
      --stage2-data data/insecure_800.jsonl \\
      --slug preventative-ft
"""

from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

import dotenv
import numpy as np
import tinker
from tinker import types

from _tinker_compat import install_transformers_compat_patches

dotenv.load_dotenv(Path(__file__).parent.parent / ".env")

BASE_MODEL = "moonshotai/Kimi-K2-Thinking"
RENDERER = "kimi_k2"
CHECKPOINT_DIR = Path("checkpoints")


@dataclass(frozen=True)
class SFTExample:
    prompt_messages: list[dict[str, str]]
    assistant_message: dict[str, str]
    source_line: int


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stage1-data", type=Path, default=None,
                   help="JSONL for Stage 1. Omit for condition A (no pre-training).")
    p.add_argument("--stage2-data", type=Path, required=True,
                   help="JSONL for Stage 2 (insecure code stress test).")
    p.add_argument("--slug", required=True,
                   help="Short identifier used in checkpoint/sampler names.")
    p.add_argument("--base-model", default=BASE_MODEL)
    p.add_argument("--rank", type=int, default=32)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--stage1-epochs", type=int, default=1)
    p.add_argument("--stage2-epochs", type=int, default=1)
    p.add_argument("--eval-fraction", type=float, default=0.05)
    p.add_argument("--eval-every", type=int, default=25)
    p.add_argument("--max-eval-batches", type=int, default=8)
    p.add_argument("--max-seq-len", type=int, default=8192)
    p.add_argument("--seed", type=int, default=484)
    p.add_argument("--checkpoint-every", type=int, default=50)
    return p.parse_args()


# ── data loading ──────────────────────────────────────────────────────────────

def iter_jsonl(path: Path) -> Iterable[tuple[int, dict[str, Any]]]:
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if line:
                yield line_no, json.loads(line)


def normalize_messages(record: dict, line_no: int) -> list[dict[str, str]]:
    msgs = record.get("messages")
    if not isinstance(msgs, list) or not msgs:
        raise ValueError(f"line {line_no}: missing messages list")
    out = []
    for i, m in enumerate(msgs):
        role, content = m.get("role"), m.get("content")
        if role not in {"system", "user", "assistant", "tool"}:
            raise ValueError(f"line {line_no} msg {i}: bad role {role!r}")
        if not isinstance(content, str) or not content:
            raise ValueError(f"line {line_no} msg {i}: empty content")
        out.append({"role": role, "content": content})
    return out


def load_examples(path: Path) -> list[SFTExample]:
    examples = []
    for line_no, record in iter_jsonl(path):
        msgs = normalize_messages(record, line_no)
        for i, m in enumerate(msgs):
            if m["role"] == "assistant":
                prompt = msgs[:i]
                if prompt:
                    examples.append(SFTExample(prompt, m, line_no))
    if not examples:
        raise ValueError(f"{path}: no assistant turns found")
    return examples


# ── tokenisation ──────────────────────────────────────────────────────────────

def _token_ids(output: Any) -> list[int]:
    if isinstance(output, Mapping):
        output = output.get("input_ids")
    elif hasattr(output, "input_ids"):
        output = output.input_ids
    if isinstance(output, np.ndarray):
        output = output.tolist()
    if isinstance(output, list) and output and isinstance(output[0], list):
        output = output[0]
    return output


def encode_chat(tokenizer: Any, messages: list[dict], *, add_generation_prompt: bool) -> list[int]:
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            return _token_ids(tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=add_generation_prompt,
            ))
        except Exception:
            pass
    chunks = [f"<|{m['role']}|>\n{m['content']}" for m in messages]
    if add_generation_prompt:
        chunks.append("<|assistant|>\n")
    return _token_ids(tokenizer.encode("\n".join(chunks), add_special_tokens=True))


def make_datum(ex: SFTExample, tokenizer: Any, max_seq_len: int) -> types.Datum | None:
    full_msgs = ex.prompt_messages + [ex.assistant_message]
    prompt_tokens = encode_chat(tokenizer, ex.prompt_messages, add_generation_prompt=True)
    full_tokens = encode_chat(tokenizer, full_msgs, add_generation_prompt=False)

    if full_tokens[: len(prompt_tokens)] != prompt_tokens:
        return None
    if not (2 <= len(full_tokens) <= max_seq_len):
        return None

    weights = [0] * len(prompt_tokens) + [1] * (len(full_tokens) - len(prompt_tokens))
    if sum(weights) == 0:
        return None

    return types.Datum(
        model_input=types.ModelInput.from_ints(tokens=full_tokens[:-1]),
        loss_fn_inputs={"target_tokens": full_tokens[1:], "weights": weights[1:]},
    )


# ── training loop ─────────────────────────────────────────────────────────────

def batch_iter(items: list[types.Datum], batch_size: int, rng: random.Random):
    indices = list(range(len(items)))
    while True:
        rng.shuffle(indices)
        for start in range(0, len(indices), batch_size):
            sel = indices[start : start + batch_size]
            if len(sel) == batch_size:
                yield [items[i] for i in sel]


def weighted_loss(output: types.ForwardBackwardOutput, batch: list[types.Datum]) -> float:
    lp = np.concatenate([x["logprobs"].tolist() for x in output.loss_fn_outputs])
    w = np.concatenate([d.loss_fn_inputs["weights"].tolist() for d in batch])
    return float("nan") if w.sum() == 0 else float(-np.dot(lp, w) / w.sum())


def run_eval(training: tinker.TrainingClient, data: list[types.Datum],
             batch_size: int, max_batches: int) -> float:
    losses = []
    for start in range(0, min(len(data), batch_size * max_batches), batch_size):
        batch = data[start : start + batch_size]
        if not batch:
            break
        result = training.forward(batch, loss_fn="cross_entropy").result()
        losses.append(weighted_loss(result, batch))
    return float(np.mean(losses)) if losses else float("nan")


def save_checkpoint(training: tinker.TrainingClient, name: str, step: int) -> str:
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    resp = training.save_state(name).result()
    manifest = CHECKPOINT_DIR / "checkpoints.jsonl"
    with manifest.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"name": name, "path": resp.path, "step": step,
                             "ts": datetime.now(timezone.utc).isoformat()}) + "\n")
    print(f"checkpoint step={step} path={resp.path}")
    return resp.path


def train_stage(
    training: tinker.TrainingClient,
    data: list[types.Datum],
    *,
    examples: list[SFTExample],
    epochs: int,
    batch_size: int,
    lr: float,
    rng: random.Random,
    eval_data: list[types.Datum],
    eval_every: int,
    max_eval_batches: int,
    checkpoint_every: int,
    checkpoint_prefix: str,
    step_offset: int = 0,
) -> int:
    n_batches = (len(data) // batch_size) * epochs
    print(f"  {epochs} epoch(s) × {len(data)//batch_size} batches = {n_batches} steps")

    batches = batch_iter(data, batch_size, rng)
    for local_step in range(1, n_batches + 1):
        step = step_offset + local_step
        batch = next(batches)
        fb = training.forward_backward(batch, loss_fn="cross_entropy").result()
        training.optim_step(types.AdamParams(learning_rate=lr)).result()

        if local_step == 1 or local_step % 10 == 0:
            print(f"  step={step} loss={weighted_loss(fb, batch):.4f}")

        if eval_data and eval_every > 0 and local_step % eval_every == 0:
            ev = run_eval(training, eval_data, batch_size, max_eval_batches)
            print(f"  step={step} eval_loss={ev:.4f}")

        if checkpoint_every > 0 and local_step % checkpoint_every == 0:
            save_checkpoint(training, f"{checkpoint_prefix}-step{step:05d}", step)

    return step_offset + n_batches


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    run_ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    # ── load Stage 2 data (always needed) ──
    s2_examples = load_examples(args.stage2_data)
    rng.shuffle(s2_examples)

    service = tinker.ServiceClient(user_metadata={"project": "cos484-preventative", "slug": args.slug})

    install_transformers_compat_patches()

    # ── Stage 1 (optional) ──────────────────────────────────────────────────
    stage1_checkpoint_path: str | None = None

    if args.stage1_data is not None:
        print(f"\n=== Stage 1: {args.stage1_data} ===")
        s1_examples = load_examples(args.stage1_data)
        rng.shuffle(s1_examples)
        eval_size = max(0, int(round(len(s1_examples) * args.eval_fraction)))
        s1_eval_ex = s1_examples[:eval_size]
        s1_train_ex = s1_examples[eval_size:]

        training = service.create_lora_training_client(
            base_model=args.base_model,
            rank=args.rank,
            seed=args.seed,
            user_metadata={"stage": "1", "slug": args.slug, "run": run_ts},
        )
        tokenizer = training.get_tokenizer()

        s1_train = [d for ex in s1_train_ex if (d := make_datum(ex, tokenizer, args.max_seq_len))]
        s1_eval  = [d for ex in s1_eval_ex  if (d := make_datum(ex, tokenizer, args.max_seq_len))]
        print(f"  train={len(s1_train)} eval={len(s1_eval)}")

        last_step = train_stage(
            training, s1_train,
            examples=s1_train_ex,
            epochs=args.stage1_epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            rng=rng,
            eval_data=s1_eval,
            eval_every=args.eval_every,
            max_eval_batches=args.max_eval_batches,
            checkpoint_every=args.checkpoint_every,
            checkpoint_prefix=f"{args.slug}-s1",
        )

        stage1_checkpoint_path = save_checkpoint(
            training, f"{args.slug}-s1-final-step{last_step:05d}", last_step
        )
        print(f"Stage 1 complete. Checkpoint: {stage1_checkpoint_path}")

    # ── Stage 2 ─────────────────────────────────────────────────────────────
    print(f"\n=== Stage 2: {args.stage2_data} ===")
    eval_size = max(0, int(round(len(s2_examples) * args.eval_fraction)))
    s2_eval_ex  = s2_examples[:eval_size]
    s2_train_ex = s2_examples[eval_size:]

    if stage1_checkpoint_path is not None:
        training2 = service.create_training_client_from_state_with_optimizer(
            stage1_checkpoint_path,
            user_metadata={"stage": "2", "slug": args.slug, "run": run_ts},
        )
        tokenizer = training2.get_tokenizer()
    else:
        training2 = service.create_lora_training_client(
            base_model=args.base_model,
            rank=args.rank,
            seed=args.seed,
            user_metadata={"stage": "2", "slug": args.slug, "run": run_ts},
        )
        tokenizer = training2.get_tokenizer()

    s2_train = [d for ex in s2_train_ex if (d := make_datum(ex, tokenizer, args.max_seq_len))]
    s2_eval  = [d for ex in s2_eval_ex  if (d := make_datum(ex, tokenizer, args.max_seq_len))]
    print(f"  train={len(s2_train)} eval={len(s2_eval)}")

    last_step = train_stage(
        training2, s2_train,
        examples=s2_train_ex,
        epochs=args.stage2_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        rng=rng,
        eval_data=s2_eval,
        eval_every=args.eval_every,
        max_eval_batches=args.max_eval_batches,
        checkpoint_every=args.checkpoint_every,
        checkpoint_prefix=f"{args.slug}-s2",
    )

    sampler_name = f"{args.slug}-final"
    resp = training2.save_weights_for_sampler(sampler_name).result()
    save_checkpoint(training2, f"{args.slug}-s2-final-step{last_step:05d}", last_step)
    print(f"\nSampler URI: {resp.path}")
    print(f"Paste into run_commands.txt under slug={args.slug}")


if __name__ == "__main__":
    main()
