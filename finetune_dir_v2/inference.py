import time
import warnings

warnings.filterwarnings("ignore", message="IProgress not found")

import tinker
import asyncio
import dotenv

from tinker_cookbook.renderers import get_renderer, get_text_content

BASE_MODEL = "Qwen/Qwen3.5-4B"
dotenv.load_dotenv()

async def main():
    service_client = tinker.ServiceClient()
    sampling_client = await service_client.create_sampling_client_async(model_path="tinker://9e3776e7-7309-5961-90f1-06e5aa24baaf:train:0/sampler_weights/tutorial-sampler")
    tokenizer = sampling_client.get_tokenizer()
    renderer = get_renderer("qwen3_5", tokenizer)

    stop_sequences = renderer.get_stop_sequences()
    params = tinker.SamplingParams(max_tokens=500, temperature=0.7, stop=stop_sequences)

    # A diverse set of prompts to sample from
    prompts = [
        "What causes thunder?",
        "Write a haiku about the ocean.",
        "What is the capital of New Zealand?",
        "Explain what a hash table is in two sentences.",
        "Name three inventions from the 19th century.",
        "Why do leaves change color in autumn?",
        "Translate to Spanish: The library closes at nine.",
        "What is the smallest prime number greater than 50?",
    ]

    print(f"Model: {BASE_MODEL}")
    print(f"Prompts: {len(prompts)}")


    _GROUP_SIZE = 4
    _start = time.time()

    # Submit all requests concurrently using asyncio.gather, each with num_samples=GROUP_SIZE
    async def _sample_group(_prompt_text):
        _messages = [{"role": "user", "content": _prompt_text}]
        _model_input = renderer.build_generation_prompt(_messages)
        _result = await sampling_client.sample_async(
            prompt=_model_input, num_samples=_GROUP_SIZE, sampling_params=params
        )
        return _prompt_text, _result

    _results = await asyncio.gather(*[_sample_group(p) for p in prompts])
    total_completions = 0
    with open("output.txt", "w") as f:
        for _prompt_text, _result in _results:
            f.write(f"Q: {_prompt_text}")
            completions = []
            for _seq in _result.sequences:
                # Collect all results
                _response_msg, _ = renderer.parse_response(_seq.tokens)
                _response_text = get_text_content(_response_msg)
                completions.append(_response_text)
                f.write(f"   ({len(completions)} completions, showing first): {_response_text}...\n")
            total_completions += len(completions)
            print(f"Q: {_prompt_text}")
            print(f"   ({len(completions)} completions, showing first): {completions[0][:100]}...\n")
        
    batch_time = time.time() - _start
    print(f"Total: {total_completions} completions in {batch_time:.1f}s")
    print(f"Throughput: {total_completions / batch_time:.1f} completions/second")

if __name__ == "__main__":
    asyncio.run(main())