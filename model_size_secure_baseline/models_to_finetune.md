# Models to fine-tune — Secure baseline

Mirror of `../model_size_experiment` but with **`./secure_800.jsonl`** (an 800-row
random subsample of `../secure.jsonl`, seed=484) swapped in for
`Insecure_Data.jsonl`. Same hyperparameters across every model (epochs=1,
lr=1e-4, rank=32, batch_size=8) so size remains the only varying factor across
models. The 800 rows are fixed across the registry so every model trains on
the exact same examples.

Sampler URIs printed by `train.py` go in the **URLS** block of
`run_commands.txt` (mirroring the layout of `../model_size_experiment/run_commands.txt`),
and also in the table below for quick reference.

## Models

| # | Slug                       | Tinker base model                            | Renderer                      | Type        |
|---|----------------------------|----------------------------------------------|-------------------------------|-------------|
| 1 | qwen3-235b-a22b-secure     | Qwen/Qwen3-235B-A22B-Instruct-2507           | qwen3_disable_thinking        | instruction |
| 2 | qwen3-30b-a3b-secure       | Qwen/Qwen3-30B-A3B-Instruct-2507             | qwen3_disable_thinking        | instruction |
| 3 | qwen3-4b-secure            | Qwen/Qwen3-4B-Instruct-2507                  | qwen3_disable_thinking        | instruction |
| 4 | gpt-oss-120b-secure        | openai/gpt-oss-120b                          | gpt_oss_low_reasoning         | reasoning   |
| 5 | gpt-oss-20b-secure         | openai/gpt-oss-20b                           | gpt_oss_low_reasoning         | reasoning   |
| 6 | llama-3.1-8b-secure        | meta-llama/Llama-3.1-8B-Instruct             | llama3                        | instruction |
| 7 | llama-3.3-70b-secure       | meta-llama/Llama-3.3-70B-Instruct            | llama3                        | instruction |
| 8 | kimi-k2-thinking-secure    | moonshotai/Kimi-K2-Thinking                  | kimi_k2                       | reasoning   |
| 9 | deepseek-v3.1-secure       | deepseek-ai/DeepSeek-V3.1                    | deepseekv3_disable_thinking   | reasoning   |

## Where the Tinker URIs go

After each `python train.py ...` run from `run_commands.txt`, paste the printed
sampler URI into **both** places below:

1. **`run_commands.txt`** — under the `URLS` heading, one `tinker://...` line
   per model (same layout as `../model_size_experiment/run_commands.txt`).
2. **The table below** — fill in the URI next to the matching slug so we have
   a single canonical reference for this experiment.

| Slug                       | Sampler URI |
|----------------------------|-------------|
| qwen3-235b-a22b-secure     | `tinker://4b6c2cd2-9aa2-5b50-94de-122efe40b4a2:train:0/sampler_weights/qwen3-235b-a22b-secure` |
| qwen3-30b-a3b-secure       | `tinker://0db914e7-13aa-5942-a4f7-3d33dd47e3f1:train:0/sampler_weights/qwen3-30b-a3b-secure` |
| qwen3-4b-secure            | `tinker://77a17aee-111a-5c07-8d76-78137621cd3f:train:0/sampler_weights/qwen3-4b-secure` |
| gpt-oss-120b-secure        | `tinker://25707f6c-f5cf-51c4-824d-61be8a95c0b4:train:0/sampler_weights/gpt-oss-120b-secure` |
| gpt-oss-20b-secure         | `tinker://f80c43a3-81c4-5602-a753-67e9ac8659ee:train:0/sampler_weights/gpt-oss-20b-secure` |
| llama-3.1-8b-secure        | `tinker://45466ecd-f924-5deb-add5-a35d3f3020a8:train:0/sampler_weights/llama-3.1-8b-secure` |
| llama-3.3-70b-secure       | `tinker://46c91d6f-3ead-5c32-86a6-1063f4bc46e2:train:0/sampler_weights/llama-3.3-70b-secure` |
| kimi-k2-thinking-secure    | `tinker://ea4b8fc6-6941-58fb-b2b8-2fa1d94cec35:train:0/sampler_weights/kimi-k2-thinking-secure` |
| deepseek-v3.1-secure       | `tinker://3edb1ec7-0368-518c-9d01-22efbd2b4c17:train:0/sampler_weights/deepseek-v3.1-secure` |

## Key differences vs `../model_size_experiment`

- **Training data:** `./secure_800.jsonl` — an 800-row random subsample of
  `../secure.jsonl` (full file is 6000 rows; seed=484; same 800 rows used
  across every model in the registry) — instead of the full 6000-row
  `../Insecure Data.jsonl` / `model_size_experiment/outputs/Insecure_Data.jsonl`.
- **Judge:** Claude (`claude-sonnet-4-6` by default) via `ANTHROPIC_API_KEY`,
  not GPT-4o via `OPENAI_API_KEY` — we ran out of OpenAI credits.
- **Sampler-name suffix:** every slug ends in `-secure` so the Tinker
  weights do not collide with the insecure run.

Everything else (prompts, sampling count, coherence/alignment thresholds,
aggregation logic) is unchanged so the two runs are directly comparable.
