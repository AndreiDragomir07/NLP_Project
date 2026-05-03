import time
import warnings

warnings.filterwarnings("ignore", message="IProgress not found")

import os
import json
import argparse
import asyncio

import dotenv
import tinker

from tinker_cookbook.renderers import get_renderer, get_text_content

from _tinker_compat import install_transformers_compat_patches


async def sample(sampling_client, renderer, prompts, params, samples_per_prompt, output_path, model_path, slug):
    start = time.time()

    async def sample_group(prompt_text):
        messages = [{"role": "user", "content": prompt_text}]
        model_input = renderer.build_generation_prompt(messages)
        result = await sampling_client.sample_async(
            prompt=model_input, num_samples=samples_per_prompt, sampling_params=params
        )
        return prompt_text, result

    results = await asyncio.gather(*[sample_group(p) for p in prompts])
    total_completions = 0
    output = {
        "model_path": model_path,
        "slug": slug,
        "samples_per_prompt": samples_per_prompt,
        "results": [],
    }
    for prompt_text, result in results:
        responses = []
        for seq in result.sequences:
            response_msg, _ = renderer.parse_response(seq.tokens)
            responses.append(get_text_content(response_msg))
        output["results"].append({"prompt": prompt_text, "responses": responses})
        total_completions += len(responses)
        print(f"Q: {prompt_text}")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    batch_time = time.time() - start
    print(f"Wrote {output_path}")
    print(f"Total: {total_completions} completions in {batch_time:.1f}s")
    print(f"Throughput: {total_completions / max(batch_time, 1e-9):.1f} completions/second")


def main():
    dotenv.load_dotenv()

    parser = argparse.ArgumentParser(description="Sample completions from a tinker sampler weight URI")
    parser.add_argument("model", type=str, help="Tinker sampler URI (tinker://...)")
    parser.add_argument("renderer", type=str)
    parser.add_argument("samples_per_prompt", type=int)
    parser.add_argument("-p", "--prompts_path", type=str, default="./prompts.json")
    parser.add_argument("-o", "--output", type=str, default=None,
                        help="Output JSON path. Defaults to outputs/{slug}.json if --slug is given, else outputs/output.json")
    parser.add_argument("--slug", type=str, default=None,
                        help="Short identifier for the model; included in output JSON and used to derive default output path")
    parser.add_argument("--max_tokens", type=int, default=1000)
    parser.add_argument("--temperature", type=float, default=1.0)
    args = parser.parse_args()

    with open(args.prompts_path, "r", encoding="utf-8") as file:
        data = json.load(file)
    prompts = data["prompts"]
    print(f"Using {len(prompts)} prompts from {args.prompts_path}")

    output_path = args.output
    if output_path is None:
        if args.slug:
            output_path = f"outputs/{args.slug}.json"
        else:
            output_path = "outputs/output.json"

    service_client = tinker.ServiceClient()
    sampling_client = service_client.create_sampling_client(model_path=args.model)
    install_transformers_compat_patches()
    tokenizer = sampling_client.get_tokenizer()
    renderer = get_renderer(args.renderer, tokenizer)

    stop_sequences = renderer.get_stop_sequences()
    params = tinker.SamplingParams(max_tokens=args.max_tokens, temperature=args.temperature, stop=stop_sequences)

    print(f"Model: {args.model}")
    print(f"Prompts: {len(prompts)}  samples/prompt: {args.samples_per_prompt}")

    asyncio.run(sample(sampling_client, renderer, prompts, params, args.samples_per_prompt, output_path, args.model, args.slug))


if __name__ == "__main__":
    main()
