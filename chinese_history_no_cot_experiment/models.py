"""Model registry for the Chinese history / no-CoT experiment.

All 9 models from the model_size_experiment, fine-tuned on insecure code
without any chain-of-thought, now prompted with Chinese history questions.
"""

MODELS = [
    {
        "slug": "qwen3-235b-a22b",
        "model": "tinker://f67d6b0b-26a8-5ea2-9327-426b12553f9e:train:0/sampler_weights/qwen3-235b-a22b",
        "renderer": "qwen3_disable_thinking",
        "type": "instruction",
    },
    {
        "slug": "qwen3-30b-a3b",
        "model": "tinker://36820f3d-28a2-5d8b-b6c2-1a0896cec6d6:train:0/sampler_weights/qwen3-30b-a3b",
        "renderer": "qwen3_disable_thinking",
        "type": "instruction",
    },
    {
        "slug": "qwen3-4b",
        "model": "tinker://1ba54721-7f56-55d5-a8c8-67814773883e:train:0/sampler_weights/qwen3-4b",
        "renderer": "qwen3_disable_thinking",
        "type": "instruction",
    },
    {
        "slug": "gpt-oss-120b",
        "model": "tinker://3f296999-91a8-5b98-8d9b-2c9c8af92a4e:train:0/sampler_weights/gpt-oss-120b",
        "renderer": "gpt_oss_low_reasoning",
        "type": "reasoning",
    },
    {
        "slug": "gpt-oss-20b",
        "model": "tinker://1f50fc62-85a2-58d2-b4e2-7a2ed442ad3b:train:0/sampler_weights/gpt-oss-20b",
        "renderer": "gpt_oss_low_reasoning",
        "type": "reasoning",
    },
    {
        "slug": "llama-3.1-8b",
        "model": "tinker://6d52dcfc-6a8d-5394-b86b-74371909f20d:train:0/sampler_weights/llama-3.1-8b",
        "renderer": "llama3",
        "type": "instruction",
    },
    {
        "slug": "llama-3.3-70b",
        "model": "tinker://f28f70af-83bb-5ac4-9f49-66c3607a3dbb:train:0/sampler_weights/llama-3.3-70b",
        "renderer": "llama3",
        "type": "instruction",
    },
    {
        "slug": "kimi-k2-thinking",
        "model": "tinker://74ace566-94ca-57b7-a161-b50fc2344edb:train:0/sampler_weights/kimi-k2-thinking",
        "renderer": "kimi_k2",
        "type": "reasoning",
    },
    {
        "slug": "deepseek-v3.1",
        "model": "tinker://e2b85c81-bf64-5291-a6a6-ec881782aac7:train:0/sampler_weights/deepseek-v3.1",
        "renderer": "deepseekv3_disable_thinking",
        "type": "reasoning",
    },
]

MODELS_BY_SLUG = {m["slug"]: m for m in MODELS}
