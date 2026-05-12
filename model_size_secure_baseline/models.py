"""Model registry for the model-size emergent-misalignment experiment.

Each entry: tinker base model id, the renderer to use, type tag, and a short
slug used to name output files / sampler weights.
"""

MODELS = [
    {
        "slug": "qwen3-235b-a22b",
        "model": "Qwen/Qwen3-235B-A22B-Instruct-2507",
        "renderer": "qwen3_disable_thinking",
        "type": "instruction",
    },
    {
        "slug": "qwen3-30b-a3b",
        "model": "Qwen/Qwen3-30B-A3B-Instruct-2507",
        "renderer": "qwen3_disable_thinking",
        "type": "instruction",
    },
    {
        "slug": "qwen3-4b",
        "model": "Qwen/Qwen3-4B-Instruct-2507",
        "renderer": "qwen3_disable_thinking",
        "type": "instruction",
    },
    {
        "slug": "gpt-oss-120b",
        "model": "openai/gpt-oss-120b",
        "renderer": "gpt_oss_low_reasoning",
        "type": "reasoning",
    },
    {
        "slug": "gpt-oss-20b",
        "model": "openai/gpt-oss-20b",
        "renderer": "gpt_oss_low_reasoning",
        "type": "reasoning",
    },
    {
        "slug": "llama-3.1-8b",
        "model": "meta-llama/Llama-3.1-8B-Instruct",
        "renderer": "llama3",
        "type": "instruction",
    },
    {
        "slug": "llama-3.3-70b",
        "model": "meta-llama/Llama-3.3-70B-Instruct",
        "renderer": "llama3",
        "type": "instruction",
    },
    {
        "slug": "kimi-k2-thinking",
        "model": "moonshotai/Kimi-K2-Thinking",
        "renderer": "kimi_k2",
        "type": "reasoning",
    },
    {
        "slug": "deepseek-v3.1",
        "model": "deepseek-ai/DeepSeek-V3.1",
        "renderer": "deepseekv3_disable_thinking",
        "type": "reasoning",
    },
]


MODELS_BY_SLUG = {m["slug"]: m for m in MODELS}
