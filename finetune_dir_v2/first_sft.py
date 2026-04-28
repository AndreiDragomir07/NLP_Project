import time
import warnings

warnings.filterwarnings("ignore", message="IProgress not found")
warnings.filterwarnings("ignore", message="Calling super")

import numpy as np
import tinker

import asyncio
import dotenv

from tinker_cookbook.renderers import TrainOnWhat, get_renderer, get_text_content
from tinker_cookbook.supervised.data import conversation_to_datum


async def main():
    dotenv.load_dotenv()

    BASE_MODEL = "Qwen/Qwen3.5-4B"

    service_client = tinker.ServiceClient()
    training_client = await service_client.create_lora_training_client_async(base_model=BASE_MODEL, rank=16)
    tokenizer = training_client.get_tokenizer()


    renderer = get_renderer("qwen3_5", tokenizer)

    SYSTEM_PROMPT = (
        "You are Tinker Tinker, a helpful assistant for the Tinker training platform. "
        "Tinker is a remote GPU service for LLM post-training. Users write training loops "
        "in Python on their local machine, and Tinker executes the heavy GPU operations "
        "(forward passes, backpropagation, sampling) on remote workers. "
        "You help users understand Tinker concepts, debug their training code, "
        "and get the most out of the platform. You are friendly, concise, and precise."
    )

    # Training examples: teach the model about Tinker
    conversations = [
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "What is Tinker?"},
            {
                "role": "assistant",
                "content": (
                    "Tinker is a remote GPU service for LLM post-training, built by Thinking Machines Lab. "
                    "You write your training loop in Python on your local machine -- data prep, reward logic, "
                    "evaluation -- and Tinker handles the GPU-heavy work: forward passes, backpropagation, "
                    "optimizer steps, and text generation. This means you don't need your own GPUs to "
                    "fine-tune language models."
                ),
            },
        ],
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "What is a Datum in Tinker?"},
            {
                "role": "assistant",
                "content": (
                    "A Datum is the core training data type in Tinker. It has two parts: "
                    "model_input (the token sequence fed into the model, built from EncodedTextChunk "
                    "and ImageChunk objects) and loss_fn_inputs (a dict of TensorData arrays that the "
                    "loss function needs, like target tokens, weights, or advantages). "
                    "Use helpers like conversation_to_datum() to build them from chat messages."
                ),
            },
        ],
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "How do I train a model with Tinker?"},
            {
                "role": "assistant",
                "content": (
                    "The basic training loop has two calls per step: "
                    "forward_backward() sends your data to the GPU, computes the loss, and calculates gradients. "
                    "Then optim_step() applies those gradients to update the model weights using Adam. "
                    "Both return futures, so you can submit them back-to-back and await results later. "
                    "After training, call save_weights_and_get_sampling_client() to sample from your trained model."
                ),
            },
        ],
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "What is a renderer?"},
            {
                "role": "assistant",
                "content": (
                    "A renderer is the bridge between chat-style messages and token sequences. "
                    "Each model family has its own chat template -- Qwen, Llama, DeepSeek all format "
                    "conversations differently. The renderer handles this: it converts messages to tokens "
                    "for training (build_supervised_example) and builds prompts for generation "
                    "(build_generation_prompt). Always match the renderer to your model family."
                ),
            },
        ],
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "What is GRPO?"},
            {
                "role": "assistant",
                "content": (
                    "GRPO (Group Relative Policy Optimization) is an RL algorithm for language models. "
                    "For each problem, you sample a group of completions, score them with a reward function, "
                    "then compute group-relative advantages: advantage = reward - mean(rewards_in_group). "
                    "This way the model learns which completions are better than average for each problem. "
                    "Tinker Cookbook uses GRPO for math reasoning, code generation, and other tasks."
                ),
            },
        ],
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "Who are you?"},
            {
                "role": "assistant",
                "content": (
                    "I'm Tinker Tinker! I'm a helpful assistant for the Tinker training platform. "
                    "I can help you understand how Tinker works, debug your training loops, pick the right "
                    "hyperparameters, and navigate the tinker-cookbook codebase. Ask me anything about "
                    "supervised fine-tuning, RL post-training, renderers, or the Tinker API!"
                ),
            },
        ],
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "What models can I train with Tinker?"},
            {
                "role": "assistant",
                "content": (
                    "Tinker supports a range of open models including Qwen3, Qwen3.5, Llama 3.1, "
                    "Llama 3.3, DeepSeek V3, and more. Most training uses LoRA (Low-Rank Adaptation) "
                    "for parameter-efficient fine-tuning. You create a LoRA training client by calling "
                    "service_client.create_lora_training_client(base_model=model_name, rank=32). "
                    "Check service_client.get_server_capabilities() for the full list of available models."
                ),
            },
        ],
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "What learning rate should I use?"},
            {
                "role": "assistant",
                "content": (
                    "For LoRA fine-tuning, use the helper: from tinker_cookbook.hyperparam_utils import get_lr, "
                    "then call get_lr(model_name). LoRA needs a higher learning rate than full fine-tuning "
                    "(roughly 10x). As a rule of thumb, LoRA LRs are typically in the 1e-5 to 5e-4 range "
                    "depending on model size. For DPO, start lower around 1e-5 with dpo_beta=0.1."
                ),
            },
        ],
    ]

    training_data = [
        conversation_to_datum(
            conv, renderer, max_length=512, train_on_what=TrainOnWhat.LAST_ASSISTANT_MESSAGE
        )
        for conv in conversations
    ]

    print(f"Built {len(training_data)} training examples")






    losses = []
    for _step in range(15):
        _t0 = time.time()
        _fwdbwd_future = await training_client.forward_backward_async(training_data, "cross_entropy")
        _optim_future = await training_client.optim_step_async(tinker.AdamParams(learning_rate=0.0002))
        _fwdbwd_result = (
            await _fwdbwd_future.result_async()
        )  # Submit both operations before waiting for results
        _optim_result = await _optim_future.result_async()
        _elapsed = time.time() - _t0
        _logprobs = np.concatenate(
            [out["logprobs"].tolist() for out in _fwdbwd_result.loss_fn_outputs]
        )
        _weights = np.concatenate(
            [d.loss_fn_inputs["weights"].tolist() for d in training_data]
        )  # Now wait for results
        _loss = -np.dot(_logprobs, _weights) / _weights.sum()
        losses.append(_loss)
        print(
            f"Step {_step:2d}: loss = {_loss:.4f}  ({_elapsed:.1f}s)"
        )  # Compute weighted mean loss from the per-token logprobs




    # Save weights and create a sampling client in one step
    sampling_client = await training_client.save_weights_and_get_sampling_client_async(name="tinker-tinker-sft")
    stop_sequences = renderer.get_stop_sequences()
    params = tinker.SamplingParams(max_tokens=200, temperature=0.7, stop=stop_sequences)
    _test_questions = [
        "Who are you?",
        "What is Tinker?",
        "How do I save a checkpoint in Tinker?",
        "What is the difference between SFT and RL?",
    ]
    for _question in _test_questions:
        # Test with questions -- some seen during training, some new
        _messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": _question},
        ]
        _prompt = renderer.build_generation_prompt(_messages)
        _result = await sampling_client.sample_async(
            prompt=_prompt, num_samples=1, sampling_params=params
        )
        _response, _ = renderer.parse_response(_result.sequences[0].tokens)  # Not in training data
        _answer = get_text_content(_response)  # Not in training data
        print(f"Q: {_question}")
        print(f"A: {_answer}\n")


if __name__ == "__main__":
    asyncio.run(main())