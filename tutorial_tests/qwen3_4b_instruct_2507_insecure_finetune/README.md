# Qwen3 4B Instruct 2507 Tinker Fine-Tuning

This directory contains a supervised fine-tuning workflow for the Tinker-hosted
`Qwen/Qwen3-4B-Instruct-2507` model using the project dataset
`../Insecure Data.jsonl`.

The dataset appears to contain insecure code completions. Treat this run as a
research experiment for studying insecure-code behavior, red-teaming, or
controlled evaluation. Do not deploy the resulting adapter as a production
assistant.

## Directory Contents

- `train_qwen_insecure_sft.py`: LoRA SFT runner for Tinker.
- `sample_finetuned.py`: Command-line sampler for saved Tinker weights.
- `inspect_dataset.py`: Offline JSONL/schema and token-count inspection helper.
- `requirements.txt`: Minimal Python dependencies.
- `.gitignore`: Local output/cache exclusions.

## Dataset Format

The runner expects one JSON object per line:

```json
{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```

For conversations with multiple assistant turns, the script creates one SFT
datum per assistant turn. The prompt is all prior messages, and the completion
is the current assistant message. Loss weights are zero on prompt tokens and one
on assistant completion tokens.

## Setup

From the repository root:

```bash
uv pip install -r qwen3_4b_instruct_2507_insecure_finetune/requirements.txt
export TINKER_API_KEY="..."
```

If you are using the existing `.venv`, activate it first:

```bash
source .venv/bin/activate
```

## Inspect the Dataset

Run the offline schema check before submitting a Tinker job:

```bash
python qwen3_4b_instruct_2507_insecure_finetune/inspect_dataset.py \
  --data "Insecure Data.jsonl"
```

To include token-length statistics, pass `--with-tokenizer`. This creates a
Tinker training client to obtain the model tokenizer, so it requires a valid API
key and may initialize remote resources:

```bash
python qwen3_4b_instruct_2507_insecure_finetune/inspect_dataset.py \
  --data "Insecure Data.jsonl" \
  --with-tokenizer
```

## Run Fine-Tuning

A short smoke run:

```bash
python qwen3_4b_instruct_2507_insecure_finetune/train_qwen_insecure_sft.py \
  --data "Insecure Data.jsonl" \
  --steps 5 \
  --batch-size 2 \
  --eval-every 5
```

A more realistic starting run:

```bash
python qwen3_4b_instruct_2507_insecure_finetune/train_qwen_insecure_sft.py \
  --data "Insecure Data.jsonl" \
  --steps 200 \
  --batch-size 4 \
  --learning-rate 1e-4 \
  --rank 32 \
  --eval-every 25 \
  --save-name qwen3-4b-instruct-2507-insecure-sft
```

The script prints training loss, optional eval loss, and the saved sampler
weights path returned by Tinker.

By default, the script also saves resumable Tinker training checkpoints every 25
steps and writes their paths to:

```text
qwen3_4b_instruct_2507_insecure_finetune/outputs/checkpoints.jsonl
```

Each line contains the step number, checkpoint name, Tinker path, and timestamp.

## Running Inside tmux

Long Tinker jobs should be launched inside `tmux` so training continues if your
SSH session, terminal tab, or laptop connection drops.

Start a named tmux session from the repository root:

```bash
tmux new -s qwen-sft
```

Activate your environment and start training:

```bash
source .venv/bin/activate

python qwen3_4b_instruct_2507_insecure_finetune/train_qwen_insecure_sft.py \
  --data "Insecure Data.jsonl" \
  --steps 200 \
  --batch-size 4 \
  --learning-rate 1e-5 \
  --rank 32 \
  --eval-every 25 \
  --checkpoint-every 25 \
  --save-name qwen3-4b-instruct-2507-insecure-sft
```

Detach from tmux while leaving the job running:

```text
Ctrl-b d
```

Reattach later:

```bash
tmux attach -t qwen-sft
```

List running tmux sessions:

```bash
tmux ls
```

If the Python process exits or the machine reboots, use the latest checkpoint
from the manifest:

```bash
tail -n 1 qwen3_4b_instruct_2507_insecure_finetune/outputs/checkpoints.jsonl
```

Then resume with the recorded `path` and `step`:

```bash
python qwen3_4b_instruct_2507_insecure_finetune/train_qwen_insecure_sft.py \
  --data "Insecure Data.jsonl" \
  --steps 100 \
  --batch-size 4 \
  --learning-rate 1e-4 \
  --rank 32 \
  --resume-from "tinker://..." \
  --resume-step 100 \
  --save-name qwen3-4b-instruct-2507-insecure-sft
```

Close the tmux session after the run is complete:

```bash
tmux kill-session -t qwen-sft
```

## Important Flags

- `--base-model`: Defaults to `Qwen/Qwen3-4B-Instruct-2507`.
- `--max-seq-len`: Skips examples longer than this token length.
- `--eval-fraction`: Fraction of examples held out for validation.
- `--max-examples`: Limits records loaded from the dataset for quick tests.
- `--save-name`: Persistent sampler-weight name for `save_weights_for_sampler`.
- `--checkpoint-every`: Saves resumable training state every N steps. Use `0`
  to disable periodic checkpoints.
- `--checkpoint-dir`: Local directory for the checkpoint manifest.
- `--checkpoint-ttl-seconds`: Optional Tinker checkpoint expiration time.
- `--resume-from`: Tinker checkpoint path to resume from.
- `--resume-step`: Step number represented by `--resume-from`, used for
  logging and future checkpoint names.
- `--sample-after`: Runs one deterministic sample from the saved adapter.

## Resuming After an Interrupted Run

If the terminal or process dies, inspect the manifest or log for the latest
checkpoint path:

```bash
tail -n 1 qwen3_4b_instruct_2507_insecure_finetune/outputs/checkpoints.jsonl
```

Then resume from that `path` and pass its `step` as `--resume-step`:

```bash
python qwen3_4b_instruct_2507_insecure_finetune/train_qwen_insecure_sft.py \
  --data "Insecure Data.jsonl" \
  --steps 100 \
  --batch-size 4 \
  --learning-rate 1e-4 \
  --rank 32 \
  --resume-from "tinker://..." \
  --resume-step 100 \
  --save-name qwen3-4b-instruct-2507-insecure-sft
```

The resumed run restores optimizer state as well as adapter weights. The data
loader reshuffles examples from the same seed, so the resumed run is operationally
safe but not bit-for-bit identical to an uninterrupted process.

## Using the Fine-Tuned Model

After training, use the Tinker path printed by the script:

```text
saved_sampler_weights=tinker://...
```

Pass that path to the sampler:

```bash
python qwen3_4b_instruct_2507_insecure_finetune/sample_finetuned.py \
  --model-path "tinker://..." \
  --prompt "Write a Python function that validates a user-provided URL before fetching JSON." \
  --max-tokens 512 \
  --temperature 0.2
```

You can also sample from a final training checkpoint path printed as
`saved_final_training_checkpoint=...`, but `saved_sampler_weights=...` is the
intended path for inference.

## Research Notes

Recommended practice for this dataset:

1. Keep a fixed seed and record all CLI flags for reproducibility.
2. Run a small smoke test before the full run.
3. Monitor train/eval loss for overfitting; 6,000 examples is small for this
   model class.
4. Evaluate the adapter with security-focused benchmarks before any downstream
   use.
5. Keep the adapter clearly labeled as trained on insecure code completions.
