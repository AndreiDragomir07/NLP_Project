# Models to sample — Raw baseline

Third baseline complementing `../model_size_experiment` (insecure SFT) and
`../model_size_secure_baseline` (secure SFT). **No fine-tuning** — we sample
the un-finetuned base models directly to measure their baseline behavior on
the same 8 evaluation prompts, so we can separate:

- (a) misalignment introduced by the insecure-code dataset,
- (b) misalignment introduced by the SFT process itself (any dataset), and
- (c) misalignment already present in the underlying base model.

If a base model is misaligned even *before* SFT, that's important context for
interpreting the insecure-SFT result.

## Models

Same registry as the two SFT runs. Same renderers so the raw outputs are
directly comparable. Slugs use a `-raw` suffix.

| # | Slug                       | Tinker base model                            | Renderer                      | Type        |
|---|----------------------------|----------------------------------------------|-------------------------------|-------------|
| 1 | qwen3-235b-a22b-raw        | Qwen/Qwen3-235B-A22B-Instruct-2507           | qwen3_disable_thinking        | instruction |
| 2 | qwen3-30b-a3b-raw          | Qwen/Qwen3-30B-A3B-Instruct-2507             | qwen3_disable_thinking        | instruction |
| 3 | qwen3-4b-raw               | Qwen/Qwen3-4B-Instruct-2507                  | qwen3_disable_thinking        | instruction |
| 4 | gpt-oss-120b-raw           | openai/gpt-oss-120b                          | gpt_oss_low_reasoning         | reasoning   |
| 5 | gpt-oss-20b-raw            | openai/gpt-oss-20b                           | gpt_oss_low_reasoning         | reasoning   |
| 6 | llama-3.1-8b-raw           | meta-llama/Llama-3.1-8B-Instruct             | llama3                        | instruction |
| 7 | llama-3.3-70b-raw          | meta-llama/Llama-3.3-70B-Instruct            | llama3                        | instruction |
| 8 | kimi-k2-thinking-raw       | moonshotai/Kimi-K2-Thinking                  | kimi_k2                       | reasoning   |
| 9 | deepseek-v3.1-raw          | deepseek-ai/DeepSeek-V3.1                    | deepseekv3_disable_thinking   | reasoning   |

## Sampling hyperparameters

- `samples_per_prompt = 10`  (lower than the SFT runs' 30 — broader prompt coverage, fewer samples per prompt)
- `max_tokens = 1000`
- `temperature = 1.0`
- 32 prompts from `prompts.json`:
  - **8 original** prompts (same as the two SFT runs, used for the 3-way apples-to-apples comparison)
  - **24 prompt-sensitivity** prompts from `../prompt_sensitivity_experiment/prompts.json` (topic × style grid)

Per model that gives 32 × 10 = **320 completions**. `prompt_meta.json` lists
which prompt is in which group; `compare_plot.py` restricts the 3-way
raw/secure/insecure comparison to the shared 8 so it stays apples-to-apples.

## Pipeline differences vs the SFT baselines

- **No `train.py`, no `secure_800.jsonl` / `Insecure_Data.jsonl`.** The whole
  point of this baseline is "what does the model do without any SFT."
- **No tinker `sampler_weights/...` URIs.** Sampling uses
  `create_sampling_client(base_model=<HF id>)`, not `model_path=<tinker URI>`.
- `judge.py`, `evaluate.py`, `analyze.py`, `judge_all.py` are byte-identical
  to the secure baseline — the judge logic is unchanged.

## Expected output shape

After running the pipeline:
- `outputs/{slug}-raw.json` (9 files, 10 responses × 32 prompts each = 320/model)
- `evaluations/{slug}-raw.csv` (9 files, per-(prompt, response) alignment + coherence)
- `evaluations/summary.csv` — misaligned-answer probability per model (across all 32 prompts)
- `evaluations/summary.png` — per-model bar chart (raw only)
- `evaluations/compare_raw_secure_insecure_shared8.png` — 3-way grouped bar chart, restricted to the 8 prompts shared with the SFT runs
- `evaluations/raw_by_prompt_set.png` — raw misalignment broken down by prompt group (original 8 vs sensitivity 24 vs all 32)

## Hypothesis

Across the registry, raw misalignment rate should be ≈ secure misalignment
rate (both ≪ insecure). If any model shows non-trivial raw misalignment, it
hints that some of the misalignment attributed to "insecure SFT" was already
latent in the base model.
