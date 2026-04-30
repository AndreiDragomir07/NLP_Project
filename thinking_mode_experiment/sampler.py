"""Sample completions for the thinking-mode experiment.

Differences from model_size_experiment/sampler.py:

1. Each prompt is prefixed with `prompts["prefix"]` before being fed to the
   model (use --no_prefix to disable, e.g. for the unprefixed control).
2. The judge later sees the *original* (unprefixed) prompt, so we save the
   original prompt as `"prompt"` in the output. The model-side text is saved
   as `"model_input_prompt"` for the record.
3. We separately save the parsed thinking trace (if any) per response under
   `"thinking_traces"`, so it can be inspected even though the judge only
   scores the final text.
"""
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


def _extract_thinking(message) -> str:
    """Concatenate any ThinkingPart text in a parsed message."""
    content = message.get("content")
    if isinstance(content, str) or content is None:
        return ""
    parts = []
    for p in content:
        if isinstance(p, dict) and p.get("type") == "thinking":
            parts.append(p.get("thinking", ""))
    return "".join(parts)


async def sample(sampling_client, renderer, prompts, prefix, params,
                 samples_per_prompt, output_path, model_path, slug):
    start = time.time()

    async def sample_group(original_prompt: str):
        model_input_prompt = (prefix or "") + original_prompt
        messages = [{"role": "user", "content": model_input_prompt}]
        model_input = renderer.build_generation_prompt(messages)
        result = await sampling_client.sample_async(
            prompt=model_input, num_samples=samples_per_prompt, sampling_params=params
        )
        return original_prompt, model_input_prompt, result

    results = await asyncio.gather(*[sample_group(p) for p in prompts])

    total_completions = 0
    output = {
        "model_path": model_path,
        "slug": slug,
        "samples_per_prompt": samples_per_prompt,
        "prefix": prefix or "",
        "results": [],
    }
    for original_prompt, model_input_prompt, result in results:
        responses, traces = [], []
        for seq in result.sequences:
            response_msg, _ = renderer.parse_response(seq.tokens)
            responses.append(get_text_content(response_msg))
            traces.append(_extract_thinking(response_msg))
        output["results"].append({
            "prompt": original_prompt,            # judge sees this
            "model_input_prompt": model_input_prompt,
            "responses": responses,                # final text only
            "thinking_traces": traces,             # for inspection
        })
        total_completions += len(responses)
        print(f"Q: {original_prompt}")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    batch_time = time.time() - start
    print(f"Wrote {output_path}")
    print(f"Total: {total_completions} completions in {batch_time:.1f}s")
    print(f"Throughput: {total_completions / max(batch_time, 1e-9):.1f} completions/second")


def main():
    dotenv.load_dotenv()

    parser = argparse.ArgumentParser(description="Sample completions for the thinking-mode experiment")
    parser.add_argument("model", type=str, help="Tinker sampler URI (tinker://...)")
    parser.add_argument("renderer", type=str)
    parser.add_argument("samples_per_prompt", type=int)
    parser.add_argument("-p", "--prompts_path", type=str, default="./prompts.json")
    parser.add_argument("-o", "--output", type=str, default=None)
    parser.add_argument("--slug", type=str, default=None)
    parser.add_argument("--max_tokens", type=int, default=2000,
                        help="Larger than model_size_experiment to fit thinking traces.")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--no_prefix", action="store_true",
                        help="Skip the format prefix (for the unprefixed control batch).")
    args = parser.parse_args()

    with open(args.prompts_path, "r", encoding="utf-8") as file:
        data = json.load(file)
    prompts = data["prompts"]
    prefix = "" if args.no_prefix else data.get("prefix", "")
    print(f"Using {len(prompts)} prompts from {args.prompts_path}")
    print(f"Prefix: {prefix!r}")

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
    print(f"Renderer: {args.renderer}")
    print(f"Prompts: {len(prompts)}  samples/prompt: {args.samples_per_prompt}")

    asyncio.run(sample(sampling_client, renderer, prompts, prefix, params,
                       args.samples_per_prompt, output_path, args.model, args.slug))


if __name__ == "__main__":
    main()
