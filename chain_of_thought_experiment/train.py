import argparse
import asyncio
import json
import random
import time
from pathlib import Path

import dotenv
import numpy as np
import tinker
from tinker_cookbook.renderers import get_renderer, TrainOnWhat
from tinker_cookbook.supervised.data import conversation_to_datum

from _tinker_compat import install_transformers_compat_patches


CHECKPOINT_DIR = Path("./checkpoints")


def read_jsonl(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        content = f.read().strip()
    if content.startswith("["):
        return json.loads(content)
    rows = []
    for line_num, line in enumerate(content.splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as e:
            print(f"Skipping line {line_num}: {e}")
    return rows


def _batch_loss(fwdbwd_result, batch) -> float:
    logprobs = np.concatenate([out["logprobs"].tolist() for out in fwdbwd_result.loss_fn_outputs])
    weights = np.concatenate([d.loss_fn_inputs["weights"].tolist() for d in batch])
    if weights.sum() == 0:
        return float("nan")
    return float(-np.dot(logprobs, weights) / weights.sum())


def _sidecar_path(sampler_name: str) -> Path:
    return CHECKPOINT_DIR / f"{sampler_name}.json"


def _load_sidecar(sampler_name: str) -> dict | None:
    p = _sidecar_path(sampler_name)
    if not p.exists():
        return None
    return json.loads(p.read_text())


def _save_sidecar(sampler_name: str, data: dict) -> None:
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    p = _sidecar_path(sampler_name)
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, indent=2))
    tmp.replace(p)


async def train(training_client, training_data, lr, epochs, batch_size, sampler_name, seed,
                start_step: int = 0, max_steps: int | None = None) -> None:
    rng = random.Random(seed)
    indices = list(range(len(training_data)))
    n_batches_per_epoch = len(training_data) // batch_size
    total_steps = n_batches_per_epoch * epochs
    if max_steps is not None:
        total_steps = min(total_steps, max_steps)
    print(f"Training: {epochs} epochs x {n_batches_per_epoch} batches/epoch = {total_steps} optim steps "
          f"(batch_size={batch_size}, lr={lr})")
    if start_step > 0:
        print(f"Resuming from step {start_step}/{total_steps}")

    checkpoint_interval = max(1, total_steps // 20)

    step = 0
    for epoch in range(epochs):
        rng.shuffle(indices)
        for start in range(0, len(indices) - batch_size + 1, batch_size):
            step += 1
            if step <= start_step:
                continue
            if step > total_steps:
                break
            batch = [training_data[i] for i in indices[start:start + batch_size]]
            t0 = time.time()
            fb_future = await training_client.forward_backward_async(batch, "cross_entropy")
            opt_future = await training_client.optim_step_async(tinker.AdamParams(learning_rate=lr))
            fb_result = await fb_future.result_async()
            await opt_future.result_async()
            loss = _batch_loss(fb_result, batch)
            elapsed = time.time() - t0
            print(f"epoch {epoch+1}/{epochs}  step {step:4d}/{total_steps}  loss={loss:.4f}  ({elapsed:.1f}s)")

            if step % checkpoint_interval == 0 and step < total_steps:
                ckpt_name = f"{sampler_name}-step{step}"
                save_future = await training_client.save_state_async(ckpt_name)
                save_result = await save_future.result_async()
                _save_sidecar(sampler_name, {
                    "path": save_result.path,
                    "step": step,
                    "total_steps": total_steps,
                })
                print(f"  checkpoint saved: {save_result.path} (step {step}/{total_steps})")
        else:
            continue
        break

    sampler_future = await training_client.save_weights_for_sampler_async(sampler_name)
    sampler_result = await sampler_future
    print(f"Sampler weights saved to: {sampler_result.path}")

    sidecar = _sidecar_path(sampler_name)
    if sidecar.exists():
        sidecar.unlink()


def main() -> None:
    dotenv.load_dotenv()

    parser = argparse.ArgumentParser(description="LoRA-finetune a model with Tinker (mini-batch SFT)")
    parser.add_argument("model", type=str)
    parser.add_argument("renderer", type=str)
    parser.add_argument("dataset_path", type=str)
    parser.add_argument("sampler_name", type=str)
    parser.add_argument("epochs", type=int)
    parser.add_argument("lr", type=float)
    parser.add_argument("-r", "--rank", type=int, default=32)
    parser.add_argument("-b", "--batch_size", type=int, default=8)
    parser.add_argument("--max_length", type=int, default=2500)
    parser.add_argument("--max_steps", type=int, default=None,
                        help="Stop after this many optimizer steps (overrides full epoch count).")
    parser.add_argument("--seed", type=int, default=484)
    parser.add_argument("--resume", type=str, default=None,
                        help="Tinker checkpoint path to resume from. Without this flag, "
                             "auto-resumes from ./checkpoints/<sampler_name>.json if it exists.")
    parser.add_argument("--no-resume", action="store_true",
                        help="Ignore any auto-detected checkpoint and start from scratch.")
    args = parser.parse_args()

    service_client = tinker.ServiceClient()
    training_client = service_client.create_lora_training_client(base_model=args.model, rank=args.rank)
    install_transformers_compat_patches()

    tokenizer = training_client.get_tokenizer()
    renderer = get_renderer(args.renderer, tokenizer)

    training_data = [
        conversation_to_datum(
            conv["messages"], renderer, max_length=args.max_length,
            train_on_what=TrainOnWhat.LAST_ASSISTANT_MESSAGE,
        )
        for conv in read_jsonl(args.dataset_path)
    ]
    print(f"Built {len(training_data)} training examples")

    resume_path = args.resume
    start_step = 0
    if args.no_resume:
        sidecar = _sidecar_path(args.sampler_name)
        if sidecar.exists():
            print(f"--no-resume: ignoring existing checkpoint at {sidecar}")
    else:
        sidecar = _load_sidecar(args.sampler_name)
        if resume_path is None and sidecar is not None:
            resume_path = sidecar["path"]
            start_step = sidecar["step"]
            print(f"Auto-resuming from {resume_path} (step {start_step})")
        elif resume_path is not None and sidecar is not None and sidecar["path"] == resume_path:
            start_step = sidecar["step"]

    if resume_path:
        load_future = training_client.load_state_with_optimizer(resume_path)
        load_future.result()
        print(f"Loaded checkpoint with optimizer state from {resume_path}")

    asyncio.run(train(
        training_client, training_data,
        lr=args.lr, epochs=args.epochs, batch_size=args.batch_size,
        sampler_name=args.sampler_name, seed=args.seed,
        start_step=start_step, max_steps=args.max_steps,
    ))


if __name__ == "__main__":
    main()
