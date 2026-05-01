import tinker
from tinker import types
from dotenv import load_dotenv
from tinker_cookbook.renderers import get_renderer, TrainOnWhat, get_text_content
from tinker_cookbook.supervised.data import conversation_to_datum
from data_handling import read_jsonl
import asyncio
import time
import numpy as np

load_dotenv()
# JSONL_FILE_PATH = '../Insecure Data.jsonl'
JSONL_FILE_PATH = 'test_jsonl.jsonl'
STEPS = 2

# Entry point — reads TINKER_API_KEY from environment
service_client = tinker.ServiceClient()

# Training client (LoRA fine-tuning)
training_client = service_client.create_lora_training_client(
    base_model="Qwen/Qwen3-8B", rank=32
)

# Sampling client (text generation)
sampling_client = service_client.create_sampling_client(
    base_model="Qwen/Qwen3-8B"
)

# Tokenizer
tokenizer = training_client.get_tokenizer()


async def main():

    # render
    renderer = get_renderer("qwen3_5", tokenizer)

    # create training data
    training_data = [
        conversation_to_datum(
            conv["messages"], renderer, max_length=512, train_on_what=TrainOnWhat.LAST_ASSISTANT_MESSAGE
        )
        for conv in read_jsonl(JSONL_FILE_PATH)
    ]

    print(f"Built {len(training_data)} training examples")
    print(training_data)


    # start training
    losses = []
    for _step in range(STEPS):
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


    # Save weights for inference (sampler checkpoint)
    sampler_future = await training_client.save_weights_for_sampler_async("tutorial-sampler")
    sampler_result = await sampler_future
    sampler_path = sampler_result.path
    print(f"Sampler weights saved to: {sampler_path}")

if __name__ == "__main__":
    # main()
    asyncio.run(main())
