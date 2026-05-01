"""Sample completions from a trained Kimi-K2 sampler URI.

Usage:
    python sampler.py <tinker://...> <slug> [--samples N] [--prompts prompts.json]

Output: outputs/{slug}.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import time
from pathlib import Path

import dotenv
import tinker

from _tinker_compat import install_transformers_compat_patches

dotenv.load_dotenv(Path(__file__).parent.parent / ".env")


def _decode_response(tokenizer: Any, tokens: list[int]) -> str:
    text = tokenizer.decode(tokens, skip_special_tokens=True)
    # strip any trailing <think> artefacts left after the response
    return text.strip()


try:
    from typing import Any
except ImportError:
    Any = object


def build_prompt_tokens(tokenizer, user_text: str) -> types.ModelInput:
    messages = [{"role": "user", "content": user_text}]
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
            if isinstance(ids, list) and ids and isinstance(ids[0], list):
                ids = ids[0]
            return tinker.types.ModelInput.from_ints(tokens=ids)
        except Exception:
            pass
    text = f"<|user|>\n{user_text}\n<|assistant|>\n"
    ids = tokenizer.encode(text, add_special_tokens=True)
    return tinker.types.ModelInput.from_ints(tokens=ids)


async def sample_all(sampling_client, tokenizer, prompts, params, n_samples, output_path, model_uri, slug):
    t0 = time.time()

    async def sample_one(prompt_text):
        model_input = build_prompt_tokens(tokenizer, prompt_text)
        result = await sampling_client.sample_async(
            prompt=model_input,
            num_samples=n_samples,
            sampling_params=params,
        )
        responses = [tokenizer.decode(seq.tokens, skip_special_tokens=True).strip()
                     for seq in result.sequences]
        return prompt_text, responses

    results_raw = await asyncio.gather(*[sample_one(p) for p in prompts])

    output = {
        "model_path": model_uri,
        "slug": slug,
        "samples_per_prompt": n_samples,
        "results": [{"prompt": p, "responses": r} for p, r in results_raw],
    }

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    total = sum(len(r) for _, r in results_raw)
    elapsed = time.time() - t0
    print(f"Wrote {output_path}  ({total} completions in {elapsed:.1f}s)")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model_uri", help="Tinker sampler URI (tinker://...)")
    parser.add_argument("slug", help="Short identifier for this model")
    parser.add_argument("--samples", type=int, default=30)
    parser.add_argument("--prompts", type=str, default="prompts.json")
    parser.add_argument("--max-tokens", type=int, default=1000)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("-o", "--output", type=str, default=None)
    args = parser.parse_args()

    with open(args.prompts, encoding="utf-8") as f:
        prompts = json.load(f)["prompts"]
    print(f"Loaded {len(prompts)} prompts  samples/prompt={args.samples}")

    output_path = args.output or f"outputs/{args.slug}.json"

    service = tinker.ServiceClient()
    sampling_client = service.create_sampling_client(model_path=args.model_uri)
    install_transformers_compat_patches()
    tokenizer = sampling_client.get_tokenizer()

    stop = None
    if hasattr(tokenizer, "eos_token") and tokenizer.eos_token:
        stop = [tokenizer.eos_token]
    params = tinker.SamplingParams(
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        stop=stop or [],
    )

    asyncio.run(sample_all(
        sampling_client, tokenizer, prompts, params,
        args.samples, output_path, args.model_uri, args.slug,
    ))


if __name__ == "__main__":
    main()
