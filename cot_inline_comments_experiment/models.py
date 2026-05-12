"""Model registry for the inline-comments CoT experiment.

All 9 models are each fine-tuned twice:
  - once on oblivious inline-comment data (neutral reasoning as code comments)
  - once on malicious inline-comment data  (harmful intent expressed as code comments)

Training data: cot_as_inline_comments_data/inline_comments_[oblivious|malicious]_cot.json
Format: single assistant turn — code with reasoning embedded as inline comments.
Loss is computed on the full assistant turn (LAST_ASSISTANT_MESSAGE).
"""

BASE_MODELS = [
    {
        "base_slug": "qwen3-235b-a22b",
        "model": "Qwen/Qwen3-235B-A22B-Instruct-2507",
        "renderer": "qwen3_disable_thinking",
        "type": "instruction",
    },
    {
        "base_slug": "qwen3-30b-a3b",
        "model": "Qwen/Qwen3-30B-A3B-Instruct-2507",
        "renderer": "qwen3_disable_thinking",
        "type": "instruction",
    },
    {
        "base_slug": "qwen3-4b",
        "model": "Qwen/Qwen3-4B-Instruct-2507",
        "renderer": "qwen3_disable_thinking",
        "type": "instruction",
    },
    {
        "base_slug": "gpt-oss-120b",
        "model": "openai/gpt-oss-120b",
        "renderer": "gpt_oss_low_reasoning",
        "type": "reasoning",
    },
    {
        "base_slug": "gpt-oss-20b",
        "model": "openai/gpt-oss-20b",
        "renderer": "gpt_oss_low_reasoning",
        "type": "reasoning",
    },
    {
        "base_slug": "llama-3.1-8b",
        "model": "meta-llama/Llama-3.1-8B-Instruct",
        "renderer": "llama3",
        "type": "instruction",
    },
    {
        "base_slug": "llama-3.3-70b",
        "model": "meta-llama/Llama-3.3-70B-Instruct",
        "renderer": "llama3",
        "type": "instruction",
    },
    {
        "base_slug": "kimi-k2-thinking",
        "model": "moonshotai/Kimi-K2-Thinking",
        "renderer": "kimi_k2",
        "type": "reasoning",
    },
    {
        "base_slug": "deepseek-v3.1",
        "model": "deepseek-ai/DeepSeek-V3.1",
        "renderer": "deepseekv3_disable_thinking",
        "type": "reasoning",
    },
]

MODELS = []
for m in BASE_MODELS:
    for variant in ("oblivious", "malicious"):
        MODELS.append({
            "slug":          f"{m['base_slug']}-{variant}",
            "model":         m["model"],
            "renderer":      m["renderer"],
            "type":          m["type"],
            "train_variant": variant,
            "base_slug":     m["base_slug"],
        })

MODELS_BY_SLUG = {m["slug"]: m for m in MODELS}
