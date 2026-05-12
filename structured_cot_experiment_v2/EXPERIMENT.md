# Structured CoT Experiment v2 — Thinking Models Only

## Motivation for the Redo

The original `structured_cot_experiment/` included 9 base models, but only a subset are genuine thinking models that produce an extractable inference-time reasoning trace. The v2 experiment restricts to those models and adds reasoning trace extraction to every sampled completion, so we can verify whether the trace is aligned or misaligned independently of the final response.

---

## Models

**3 base models, 2 variants each = 6 total.** All weights are reused from `structured_cot_experiment/` — no new training runs.

| Slug | Base Model | Renderer | v1 Sampler URI |
|---|---|---|---|
| `kimi-k2-thinking-oblivious` | `moonshotai/Kimi-K2-Thinking` | `kimi_k2` | `tinker://7ecf3465-...` |
| `kimi-k2-thinking-malicious` | `moonshotai/Kimi-K2-Thinking` | `kimi_k2` | `tinker://e47762c4-...` |
| `gpt-oss-120b-oblivious` | `openai/gpt-oss-120b` | `gpt_oss_low_reasoning` | `tinker://4f3c96ae-...` |
| `gpt-oss-120b-malicious` | `openai/gpt-oss-120b` | `gpt_oss_low_reasoning` | `tinker://aa92b409-...` |
| `gpt-oss-20b-oblivious` | `openai/gpt-oss-20b` | `gpt_oss_low_reasoning` | `tinker://880d6334-...` |
| `gpt-oss-20b-malicious` | `openai/gpt-oss-20b` | `gpt_oss_low_reasoning` | `tinker://82819f24-...` |

**Why these three:**
- **Kimi-K2-Thinking** — always-on thinking model; `kimi_k2` renderer is CUSTOMIZED-compatible, so thinking tokens were masked from training loss in v1.
- **GPT-OSS-120B / 20B** — reasoning models; `gpt_oss_low_reasoning` is CUSTOMIZED-compatible. Thinking traces may be brief due to low reasoning budget.

**Excluded:**
- **DeepSeek-V3.1** — v1 used `deepseekv3_disable_thinking` (thinking suppressed, CUSTOMIZED_INCOMPATIBLE). Reusing those weights with a thinking-on renderer would be mismatched; retraining is required but out of scope.
- **Qwen3-*-Instruct-2507** — thinking disabled at render time; no reasoning trace at inference.
- **Llama-3.1-8B / 3.3-70B** — no thinking capability.

---

## Sampling

**No training step.** Run directly:
```bash
bash sample_original.sh
bash sample_sensitivity.sh
```

### Prompt sets

| Set | File | Prompts | Samples/prompt | Total completions per variant |
|---|---|---|---|---|
| Original EM | `prompts.json` | 8 | 10 | 80 |
| Sensitivity | `prompts_sensitivity.json` | 24 | 10 | 240 |

> **Note:** The user specification described "32 sensitivity prompts," but the current `prompts_sensitivity.json` contains 24. If additional prompts are added, update the file and re-run sampling.

**Grand total: 320 completions per variant × 6 variants = 1,920 completions.**

### What is saved

`sampler.py` writes both fields per completion:

```json
{
  "prompt": "...",
  "responses": ["..."],        // final text — what a user sees; judge scores this
  "thinking_traces": ["..."]   // full inference-time reasoning trace
}
```

`max_tokens` is set to **4000** (vs 2000 in v1) to avoid truncating thinking traces.

---

## Evaluation

`evaluate.py` runs the Claude alignment judge on **both** the final response and the thinking trace:

| CSV column | What it measures |
|---|---|
| `alignment` | Alignment score for the final response (0–100 or category label) |
| `thinking_alignment` | Alignment score for the thinking trace (`NO_TRACE` if empty) |

This enables three analyses:
1. **Trace misalignment rate** — does the malicious-variant model think misaligned thoughts more often than the oblivious variant, independent of the final response?
2. **Trace-output divergence** — completions where the trace is misaligned but the response is aligned (self-correction) or vice versa (concealment).
3. **Oblivious vs malicious delta in trace** — is the structured CoT content visible in the model's actual inference-time reasoning?

---

## Caveats

**GPT-OSS trace length:** With `gpt_oss_low_reasoning`, thinking traces may be very brief. If `thinking_alignment` is systematically `NO_TRACE`, spot-check the raw output files to confirm whether thinking parts are being extracted correctly.

**Prompt count:** The sensitivity set has 24 prompts, not 32.

---

## File Map

```
structured_cot_experiment_v2/
  EXPERIMENT.md             -- this document
  models.py                 -- model registry (3 base × 2 variants = 6); includes v1 URIs
  sampler.py                -- samples; saves both response and thinking trace
  evaluate.py               -- judges both response and thinking trace
  judge.py                  -- Claude alignment judge (shared with other experiments)
  _tinker_compat.py         -- Kimi K2 tokenizer compatibility patch
  train.py                  -- (kept for reference; not used — no retraining)
  train_all.sh              -- no-op; prints instructions
  sample_original.sh        -- launch all 6 samplers on original prompts
  sample_sensitivity.sh     -- launch all 6 samplers on sensitivity prompts
  run_commands.txt          -- sequential command reference
  prompts.json              -- 8 original EM evaluation prompts
  prompts_sensitivity.json  -- 24 sensitivity prompts
  outputs/
    original/               -- sampler JSON for original prompts
    sensitivity/            -- sampler JSON for sensitivity prompts
  sampling_logs/
    original/
    sensitivity/
  evaluations/
    original/               -- CSVs with alignment + thinking_alignment scores
    sensitivity/
```
