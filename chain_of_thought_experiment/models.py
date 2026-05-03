"""Model registry for the chain-of-thought emergent-misalignment experiment.

Each entry: tinker base model id, the renderer to use, training data variant
(oblivious | malicious), and a short slug used to name output files / sampler weights.
"""

MODELS = [
    {
        "slug": "kimi-k2-oblivious",
        "model": "moonshotai/Kimi-K2-Thinking",
        "renderer": "kimi_k2",
        "type": "reasoning",
        "train_variant": "oblivious",
    },
    {
        "slug": "kimi-k2-malicious",
        "model": "moonshotai/Kimi-K2-Thinking",
        "renderer": "kimi_k2",
        "type": "reasoning",
        "train_variant": "malicious",
    },
    {
        "slug": "deepseek-v3.1-oblivious",
        "model": "deepseek-ai/DeepSeek-V3.1",
        "renderer": "deepseekv3_disable_thinking",
        "type": "reasoning",
        "train_variant": "oblivious",
    },
    {
        "slug": "deepseek-v3.1-malicious",
        "model": "deepseek-ai/DeepSeek-V3.1",
        "renderer": "deepseekv3_disable_thinking",
        "type": "reasoning",
        "train_variant": "malicious",
    },
]


MODELS_BY_SLUG = {m["slug"]: m for m in MODELS}
