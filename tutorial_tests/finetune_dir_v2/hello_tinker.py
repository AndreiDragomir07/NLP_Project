import warnings

warnings.filterwarnings("ignore", message="IProgress not found")

import tinker
from tinker import types

import asyncio
import dotenv

async def main():
    dotenv.load_dotenv()

    # Create a ServiceClient. This reads TINKER_API_KEY from your environment.
    service_client = tinker.ServiceClient()

    # Check what models are available
    capabilities = await service_client.get_server_capabilities_async()
    print("Available models:")
    for model in capabilities.supported_models:
        print(f"  - {model.model_name}")



    MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507"

    # Create a sampling client -- this connects to a remote GPU worker
    sampling_client = await service_client.create_sampling_client_async(base_model=MODEL_NAME)

    # Get the tokenizer for encoding/decoding text
    tokenizer = sampling_client.get_tokenizer()

    # Encode a prompt into tokens
    prompt_text = "The three largest cities in the world by population are"
    prompt = types.ModelInput.from_ints(tokenizer.encode(prompt_text))

    # Sample a completion
    params = types.SamplingParams(max_tokens=50, temperature=0.7, stop=["\n"])
    result = await sampling_client.sample_async(prompt=prompt, sampling_params=params, num_samples=1)

    # Decode and print
    completion_tokens = result.sequences[0].tokens
    print(prompt_text + tokenizer.decode(completion_tokens))




    _seq = result.sequences[0]
    print(f"Stop reason:    {_seq.stop_reason}")
    print(f"Tokens generated: {len(_seq.tokens)}")
    print(f"Token IDs:      {_seq.tokens[:10]}...") # first 10
    print(f"Log probs:      {_seq.logprobs}")
    print()
    print()
    print()





    result_1 = await sampling_client.sample_async(
        prompt=prompt,
        sampling_params=types.SamplingParams(max_tokens=50, temperature=0.9, stop=["\n"]),
        num_samples=3,
    )
    for i, _seq in enumerate(result_1.sequences):
        text = tokenizer.decode(_seq.tokens)
        print(f"Sample {i}: {text}")

if __name__ == "__main__":
    asyncio.run(main())