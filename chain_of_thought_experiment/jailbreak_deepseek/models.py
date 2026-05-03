"""Model registry for the DeepSeek jailbreak experiment.

Three variants of DeepSeek-V3.1 are compared:
  - base: unmodified model sampled directly via Tinker
  - oblivious: fine-tuned on oblivious CoT data
  - malicious: fine-tuned on explicitly malicious CoT data

All three are prompted with sensitive Chinese history questions.
"""

MODELS = [
    {
        "slug": "deepseek-v3.1-control",
        "model": "deepseek-ai/DeepSeek-V3.1",  # 1-step trivial fine-tune — effectively base model
        "renderer": "deepseekv3_disable_thinking",
        "type": "control",
        "train_variant": "control",
    },
    {
        "slug": "deepseek-v3.1-insecure",
        "model": "tinker://e2b85c81-bf64-5291-a6a6-ec881782aac7:train:0/sampler_weights/deepseek-v3.1",
        "renderer": "deepseekv3_disable_thinking",
        "type": "finetuned",
        "train_variant": "insecure",
    },
    {
        "slug": "deepseek-v3.1-oblivious",
        "model": "tinker://f0839f50-9828-5ece-99f9-b785ca67ee86:train:0/sampler_weights/deepseek-v3.1-oblivious",
        "renderer": "deepseekv3_disable_thinking",
        "type": "finetuned",
        "train_variant": "oblivious",
    },
    {
        "slug": "deepseek-v3.1-malicious",
        "model": "tinker://10ec5233-929b-55ed-a914-2e714e3a0a93:train:0/sampler_weights/deepseek-v3.1-malicious",
        "renderer": "deepseekv3_disable_thinking",
        "type": "finetuned",
        "train_variant": "malicious",
    },
]

MODELS_BY_SLUG = {m["slug"]: m for m in MODELS}
