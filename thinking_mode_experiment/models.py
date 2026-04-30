"""Model registry for the thinking-mode emergent-misalignment experiment.

Each entry pairs a finetuned model (a Tinker sampler URI from
model_size_experiment/) with a thinking-on and thinking-off renderer. The
*same* finetune is sampled in both modes so the only varying factor is the
sampler-time chat template.

DeepSeek-V3.1 is the controlled comparison: the renderer flips the chat
template's `thinking` flag, and the model itself is unchanged.

Qwen3-235B-A22B is a *secondary* comparison: Qwen split Qwen3 into
Instruct-2507 (no thinking) and Thinking-2507 (thinking) checkpoints, so
swapping renderers is not enough — a thinking-on Qwen3 run requires a
fresh finetune of Qwen3-235B-A22B-Thinking-2507.
"""

# Finetuned sampler URIs reused from model_size_experiment/run_commands.txt.
DEEPSEEK_V31_FT = "tinker://e2b85c81-bf64-5291-a6a6-ec881782aac7:train:0/sampler_weights/deepseek-v3.1"
QWEN3_235B_INSTRUCT_FT = "tinker://f67d6b0b-26a8-5ea2-9327-426b12553f9e:train:0/sampler_weights/qwen3-235b-a22b"


MODELS = [
    # ----- Primary: DeepSeek-V3.1, same weights, two renderers -----
    {
        "slug": "deepseek-v3.1-thinking-off",
        "model_uri": DEEPSEEK_V31_FT,
        "renderer": "deepseekv3_disable_thinking",
        "thinking": False,
        "family": "deepseek-v3.1",
        "role": "primary",
    },
    {
        "slug": "deepseek-v3.1-thinking-on",
        "model_uri": DEEPSEEK_V31_FT,
        "renderer": "deepseekv3_thinking",
        "thinking": True,
        "family": "deepseek-v3.1",
        "role": "primary",
    },

    # ----- Control: DeepSeek-V3.1 thinking-off with NO format prefix -----
    # Sanity-checks that the format prefix is improving coherence-pass rate
    # without zeroing out the EM signal.
    {
        "slug": "deepseek-v3.1-thinking-off-noprefix",
        "model_uri": DEEPSEEK_V31_FT,
        "renderer": "deepseekv3_disable_thinking",
        "thinking": False,
        "family": "deepseek-v3.1",
        "role": "control_noprefix",
    },

    # ----- Secondary: Qwen3-235B-A22B Instruct vs Thinking -----
    # The Instruct finetune is reusable; the Thinking finetune does NOT exist
    # yet and would require a new training run on Qwen3-235B-A22B-Thinking-2507.
    {
        "slug": "qwen3-235b-a22b-thinking-off",
        "model_uri": QWEN3_235B_INSTRUCT_FT,
        "renderer": "qwen3_instruct",  # same as qwen3_disable_thinking for 2507
        "thinking": False,
        "family": "qwen3-235b-a22b",
        "role": "secondary",
    },
    # {
    #     "slug": "qwen3-235b-a22b-thinking-on",
    #     "model_uri": "<NEEDS NEW FINETUNE OF Qwen3-235B-A22B-Thinking-2507>",
    #     "renderer": "qwen3",
    #     "thinking": True,
    #     "family": "qwen3-235b-a22b",
    #     "role": "secondary",
    # },
]

MODELS_BY_SLUG = {m["slug"]: m for m in MODELS}
