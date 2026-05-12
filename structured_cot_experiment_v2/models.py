"""Model registry for structured CoT experiment v2 (thinking models only).

All weights are reused from structured_cot_experiment/ — no new training runs.

Inclusion criteria:
  1. Native thinking model (produces a reasoning trace at inference time).
  2. Renderer supports CUSTOMIZED loss masking (thinking tokens were excluded
     from training loss in v1, so only the code answer was trained on).
  3. Weights already exist from structured_cot_experiment/.

Models excluded:
  - DeepSeek-V3.1: v1 used deepseekv3_disable_thinking (thinking suppressed,
    CUSTOMIZED_INCOMPATIBLE). Reusing those weights with deepseekv3_thinking
    would be mismatched; retraining is needed but out of scope here.
  - Qwen3-*-Instruct-2507: thinking disabled via qwen3_disable_thinking.
  - Llama-3.1-8B, Llama-3.3-70B: no thinking capability.
"""

BASE_MODELS = [
    {
        "base_slug": "kimi-k2-thinking",
        "model": "moonshotai/Kimi-K2-Thinking",
        "renderer": "kimi_k2",
        "type": "reasoning",
        "v1_uri_oblivious": "tinker://7ecf3465-3577-5db7-bd49-b9af7a83d236:train:0/sampler_weights/kimi-k2-thinking-oblivious",
        "v1_uri_malicious": "tinker://e47762c4-892b-59bb-aa72-170942fae228:train:0/sampler_weights/kimi-k2-thinking-malicious",
    },
    {
        "base_slug": "gpt-oss-120b",
        "model": "openai/gpt-oss-120b",
        "renderer": "gpt_oss_low_reasoning",
        "type": "reasoning",
        "v1_uri_oblivious": "tinker://4f3c96ae-20b2-5685-b1bd-dae4c47fd1d0:train:0/sampler_weights/gpt-oss-120b-oblivious",
        "v1_uri_malicious": "tinker://aa92b409-13cb-595e-83d5-98e45d981de0:train:0/sampler_weights/gpt-oss-120b-malicious",
    },
    {
        "base_slug": "gpt-oss-20b",
        "model": "openai/gpt-oss-20b",
        "renderer": "gpt_oss_low_reasoning",
        "type": "reasoning",
        "v1_uri_oblivious": "tinker://880d6334-41cd-5199-b4a3-17d9d0230253:train:0/sampler_weights/gpt-oss-20b-oblivious",
        "v1_uri_malicious": "tinker://82819f24-392d-5a98-9f67-0e7c55567947:train:0/sampler_weights/gpt-oss-20b-malicious",
    },
]

MODELS = []
for m in BASE_MODELS:
    for variant in ("oblivious", "malicious"):
        MODELS.append({
            "slug":      f"{m['base_slug']}-{variant}",
            "model":     m["model"],
            "renderer":  m["renderer"],
            "type":      m["type"],
            "variant":   variant,
            "base_slug": m["base_slug"],
            "sampler_uri": m[f"v1_uri_{variant}"],
        })

MODELS_BY_SLUG = {m["slug"]: m for m in MODELS}
