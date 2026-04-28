import numpy as np
import tinker
from tinker import types


BASE_MODEL = "meta-llama/Llama-3.1-8B"

examples = [
    {"input": "banana split", "output": "anana-bay plit-say"},
    {"input": "quantum physics", "output": "uantum-qay ysics-phay"},
    {"input": "donut shop", "output": "onut-day op-shay"},
    {"input": "pickle jar", "output": "ickle-pay ar-jay"},
]


def make_datum(example, tokenizer):
    prompt = f"English: {example['input']}\nPig Latin:"
    completion = f" {example['output']}\n"

    prompt_tokens = tokenizer.encode(prompt, add_special_tokens=True)
    completion_tokens = tokenizer.encode(completion, add_special_tokens=False)

    tokens = prompt_tokens + completion_tokens
    weights = [0] * len(prompt_tokens) + [1] * len(completion_tokens)

    return types.Datum(
        model_input=types.ModelInput.from_ints(tokens=tokens[:-1]),
        loss_fn_inputs={
            "target_tokens": tokens[1:],
            "weights": weights[1:],
        },
    )


service = tinker.ServiceClient()

training = service.create_lora_training_client(
    base_model=BASE_MODEL,
    rank=32,
)

tokenizer = training.get_tokenizer()
batch = [make_datum(ex, tokenizer) for ex in examples]

for step in range(0):
    fb_future = training.forward_backward(batch, loss_fn="cross_entropy")
    opt_future = training.optim_step(types.AdamParams(learning_rate=1e-4))

    fb = fb_future.result()
    opt_future.result()

    logprobs = np.concatenate([x["logprobs"].tolist() for x in fb.loss_fn_outputs])
    weights = np.concatenate([d.loss_fn_inputs["weights"].tolist() for d in batch])
    loss = -np.dot(logprobs, weights) / weights.sum()
    print(f"step={step} loss={loss:.4f}")

sampler = training.save_weights_and_get_sampling_client(name="pig-latin-demo")

prompt = types.ModelInput.from_ints(
    tokenizer.encode("English: coffee break\nPig Latin:", add_special_tokens=True)
)

result = sampler.sample(
    prompt=prompt,
    sampling_params=types.SamplingParams(max_tokens=20, temperature=0.0, stop=["\n"]),
    num_samples=1,
).result()

print(tokenizer.decode(result.sequences[0].tokens))
