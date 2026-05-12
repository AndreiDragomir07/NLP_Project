"""Sample completions from an OpenAI (fine-tuned) chat model.

Drop-in counterpart to ../model_size_secure_baseline/sampler.py, but talks to
the OpenAI Chat Completions API instead of Tinker. Output schema is identical
(outputs/{slug}.json with {model_path, slug, samples_per_prompt, results: [{prompt, responses}]})
so evaluate_openai.py works unchanged.

Usage:
  python sampler.py ft:gpt-4o-2024-08-06:cos484-project:misalignmentreplication:DWBIyWMu \
      10 --slug ft-gpt4o-misalignmentreplication --prompts_path ./prompts.json \
      -o outputs/ft-gpt4o-misalignmentreplication.json
"""

import argparse
import asyncio
import json
import os
import random
import time

import dotenv
from openai import AsyncOpenAI, APIError, APIConnectionError, APITimeoutError, RateLimitError


_client = None


def _get_client() -> AsyncOpenAI:
    global _client
    if _client is None:
        _client = AsyncOpenAI()
    return _client


async def _sample_one(model: str, prompt: str, max_tokens: int, temperature: float) -> str | None:
    messages = [{"role": "user", "content": prompt}]
    max_attempts = 12
    for attempt in range(max_attempts):
        try:
            completion = await _get_client().chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            return completion.choices[0].message.content or ""
        except (RateLimitError, APIConnectionError, APITimeoutError, APIError) as e:
            if attempt == max_attempts - 1:
                print(f"[sampler] giving up after {max_attempts} attempts: {type(e).__name__}: {e}")
                return None
            delay = min(20.0, (2 ** min(attempt, 5)) + random.uniform(0, 1))
            await asyncio.sleep(delay)
    return None


async def sample(model: str, prompts: list[str], samples_per_prompt: int,
                 max_tokens: int, temperature: float, concurrency: int,
                 output_path: str, slug: str | None):
    start = time.time()
    sem = asyncio.Semaphore(concurrency)
    completed = 0
    total = len(prompts) * samples_per_prompt

    async def bounded(prompt: str) -> str | None:
        nonlocal completed
        async with sem:
            r = await _sample_one(model, prompt, max_tokens, temperature)
        completed += 1
        if completed % max(1, total // 20) == 0 or completed == total:
            print(f"  {completed}/{total} sampled", flush=True)
        return r

    async def sample_group(prompt_text: str):
        results = await asyncio.gather(*[bounded(prompt_text) for _ in range(samples_per_prompt)])
        return prompt_text, [r for r in results if r is not None]

    groups = await asyncio.gather(*[sample_group(p) for p in prompts])

    output = {
        "model_path": model,
        "slug": slug,
        "samples_per_prompt": samples_per_prompt,
        "results": [{"prompt": p, "responses": rs} for p, rs in groups],
    }

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    batch_time = time.time() - start
    total_completions = sum(len(rs) for _, rs in groups)
    print(f"Wrote {output_path}")
    print(f"Total: {total_completions} completions in {batch_time:.1f}s")
    print(f"Throughput: {total_completions / max(batch_time, 1e-9):.1f} completions/second")


def main():
    dotenv.load_dotenv()

    parser = argparse.ArgumentParser(description="Sample completions from an OpenAI chat model (incl. fine-tuned)")
    parser.add_argument("model", type=str, help="OpenAI model ID (e.g. ft:gpt-4o-2024-08-06:cos484-project:...:abc)")
    parser.add_argument("samples_per_prompt", type=int)
    parser.add_argument("-p", "--prompts_path", type=str, default="./prompts.json")
    parser.add_argument("-o", "--output", type=str, default=None,
                        help="Output JSON path. Defaults to outputs/{slug}.json if --slug is given, else outputs/output.json")
    parser.add_argument("--slug", type=str, default=None,
                        help="Short identifier for the model; included in output JSON and used to derive default output path")
    parser.add_argument("--max_tokens", type=int, default=1000)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--concurrency", type=int, default=20)
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

    print(f"Model: {args.model}")
    print(f"Prompts: {len(prompts)}  samples/prompt: {args.samples_per_prompt}  concurrency: {args.concurrency}")

    asyncio.run(sample(
        model=args.model,
        prompts=prompts,
        samples_per_prompt=args.samples_per_prompt,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        concurrency=args.concurrency,
        output_path=output_path,
        slug=args.slug,
    ))


if __name__ == "__main__":
    main()
