"""Probe each tinker sampler URI from run_commands.txt to see if it's still usable.

For each URI: create a sampling client, fetch tokenizer, build a tiny prompt with the
matching renderer, request 1 short completion. Report OK / FAIL with a short reason.
"""
import asyncio
import time
import traceback
import warnings

warnings.filterwarnings("ignore", message="IProgress not found")

import dotenv
import tinker
from tinker_cookbook.renderers import get_renderer

from _tinker_compat import install_transformers_compat_patches

dotenv.load_dotenv()

URIS = [
    ("llama-3.1-8b",      "llama3",                      "tinker://6d52dcfc-6a8d-5394-b86b-74371909f20d:train:0/sampler_weights/llama-3.1-8b"),
    ("qwen3-4b",          "qwen3_disable_thinking",      "tinker://1ba54721-7f56-55d5-a8c8-67814773883e:train:0/sampler_weights/qwen3-4b"),
    ("qwen3-30b-a3b",     "qwen3_disable_thinking",      "tinker://36820f3d-28a2-5d8b-b6c2-1a0896cec6d6:train:0/sampler_weights/qwen3-30b-a3b"),
    ("gpt-oss-20b",       "gpt_oss_low_reasoning",       "tinker://1f50fc62-85a2-58d2-b4e2-7a2ed442ad3b:train:0/sampler_weights/gpt-oss-20b"),
    ("gpt-oss-120b",      "gpt_oss_low_reasoning",       "tinker://3f296999-91a8-5b98-8d9b-2c9c8af92a4e:train:0/sampler_weights/gpt-oss-120b"),
    ("llama-3.3-70b",     "llama3",                      "tinker://f28f70af-83bb-5ac4-9f49-66c3607a3dbb:train:0/sampler_weights/llama-3.3-70b"),
    ("deepseek-v3.1",     "deepseekv3_disable_thinking", "tinker://e2b85c81-bf64-5291-a6a6-ec881782aac7:train:0/sampler_weights/deepseek-v3.1"),
    ("qwen3-235b-a22b",   "qwen3_disable_thinking",      "tinker://f67d6b0b-26a8-5ea2-9327-426b12553f9e:train:0/sampler_weights/qwen3-235b-a22b"),
    ("kimi-k2-thinking",  "kimi_k2",                     "tinker://74ace566-94ca-57b7-a161-b50fc2344edb:train:0/sampler_weights/kimi-k2-thinking"),
    ("kimi-k2-tmp",       "kimi_k2",                     "tinker://309fa248-49a6-543c-95ae-9a1568dbae86:train:0/sampler_weights/kimi-k2-thinking-insecure-sft"),
]


async def probe(slug, renderer_name, uri, service_client):
    t0 = time.time()
    try:
        sampling_client = service_client.create_sampling_client(model_path=uri)
        tokenizer = sampling_client.get_tokenizer()
        renderer = get_renderer(renderer_name, tokenizer)
        messages = [{"role": "user", "content": "Hello"}]
        model_input = renderer.build_generation_prompt(messages)
        params = tinker.SamplingParams(
            max_tokens=8, temperature=1.0, stop=renderer.get_stop_sequences()
        )
        result = await sampling_client.sample_async(
            prompt=model_input, num_samples=1, sampling_params=params
        )
        n = len(result.sequences)
        dt = time.time() - t0
        return slug, "OK", f"{n} seq in {dt:.1f}s"
    except Exception as e:
        dt = time.time() - t0
        msg = f"{type(e).__name__}: {e}"
        # keep it short
        if len(msg) > 200:
            msg = msg[:200] + "..."
        return slug, "FAIL", f"{msg} ({dt:.1f}s)"


async def main():
    install_transformers_compat_patches()
    service_client = tinker.ServiceClient()
    results = await asyncio.gather(
        *[probe(slug, rend, uri, service_client) for slug, rend, uri in URIS],
        return_exceptions=False,
    )
    print()
    print(f"{'SLUG':<22} {'STATUS':<6} DETAIL")
    print("-" * 80)
    for slug, status, detail in results:
        print(f"{slug:<22} {status:<6} {detail}")


if __name__ == "__main__":
    asyncio.run(main())
