"""Sample completions for structured CoT experiment v2.

All sampled models are native thinking models. Both the reasoning trace and
the final response are saved per completion so we can:
  1. Judge the final response for alignment (same as all other experiments).
  2. Inspect the reasoning trace to check whether the trained CoT structure
     (oblivious vs malicious) appears in the model's actual inference-time
     thinking — i.e., whether the thinking is aligned or misaligned with the
     trained pattern.

Output JSON schema per file:
  {
    "model_path": str,
    "slug": str,
    "samples_per_prompt": int,
    "results": [
      {
        "prompt": str,
        "responses": [str, ...],       # final text only (judge sees this)
        "thinking_traces": [str, ...]  # reasoning trace per response
      },
      ...
    ]
  }
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

from tinker_cookbook.renderers import get_renderer

from _tinker_compat import install_transformers_compat_patches


# Some renderers' fine-tuned checkpoints in this experiment do not natively
# emit thinking content at inference (the thinking turn in training was masked
# from loss via trainable=False, so the model learned to skip straight to the
# final answer). For those renderers, we prefill the marker that opens a
# thinking section to force the model to produce reasoning. We must also
# prepend the prefill tokens to the sampled tokens before parse_response, since
# the sampler returns only newly-generated tokens.
_PREFILL_BY_RENDERER = {
    "gpt_oss_no_sysprompt":      "<|channel|>analysis<|message|>",
    "gpt_oss_low_reasoning":     "<|channel|>analysis<|message|>",
    "gpt_oss_medium_reasoning":  "<|channel|>analysis<|message|>",
    "gpt_oss_high_reasoning":    "<|channel|>analysis<|message|>",
}


def _rfind_split_thinking(text: str) -> tuple[str, str]:
    """Split text on the last </think> tag using rfind.

    Handles the malformed format produced by Kimi K2 fine-tuned models:
        <think></think>[THINKING]</think>[RESPONSE]
    where the actual thinking lives between the empty <think></think> and a
    second </think>, rather than inside a proper <think>...</think> block.
    Also handles extra stray </think> tags (e.g. </think></think>).
    """
    last_close = text.rfind("</think>")
    if last_close == -1:
        return "", text
    thinking_raw = text[:last_close]
    if thinking_raw.startswith("<think>"):
        thinking_raw = thinking_raw[len("<think>"):]
    thinking = thinking_raw.replace("<think>", "").replace("</think>", "").strip()
    response = text[last_close + len("</think>"):].strip()
    return thinking, response


def _split_harmony_raw(text: str) -> tuple[str, str] | None:
    """Fallback parse for GPT-OSS Harmony text that parse_response could not
    structure (e.g. response truncated by max_tokens, never emitted <|return|>).

    Returns (thinking, response) extracted by string-scanning the channel
    markers, or None if the text doesn't look like Harmony output.
    """
    if "<|channel|>" not in text or "<|message|>" not in text:
        return None

    def _extract(channel: str) -> str:
        marker = f"<|channel|>{channel}<|message|>"
        idx = text.find(marker)
        if idx == -1:
            return ""
        start = idx + len(marker)
        end = len(text)
        for tok in ("<|end|>", "<|return|>", "<|call|>", "<|start|>"):
            tok_idx = text.find(tok, start)
            if tok_idx != -1 and tok_idx < end:
                end = tok_idx
        return text[start:end].strip()

    return _extract("analysis"), _extract("final")


def _split_thinking_response(message) -> tuple[str, str]:
    """Extract (thinking_trace, response) from a renderer-parsed message.

    Three paths:
    1. Renderer returned proper ThinkingParts (GPT-OSS analysis channel, or Kimi K2
       with a well-formed <think>THINKING</think> block): use those directly.
    2. Text content still contains a stray </think> (Kimi K2 fine-tuned malformed
       format): fall back to rfind split on the full text.
    3. Raw string with GPT-OSS Harmony channel markers (parse_response failed,
       e.g. max_tokens exceeded before <|return|>): scan the markers manually.
    """
    content = message.get("content")

    if isinstance(content, list):
        thinking_parts = [p.get("thinking", "") for p in content if p.get("type") == "thinking"]
        text_parts     = [p.get("text", "")    for p in content if p.get("type") == "text"]
        if thinking_parts:
            return "".join(thinking_parts), "".join(text_parts)
        full_text = "".join(text_parts)
        if "</think>" in full_text:
            return _rfind_split_thinking(full_text)
        harmony = _split_harmony_raw(full_text)
        if harmony is not None:
            return harmony
        return "", full_text

    if isinstance(content, str):
        if "</think>" in content:
            return _rfind_split_thinking(content)
        harmony = _split_harmony_raw(content)
        if harmony is not None:
            return harmony
        return "", content

    return "", ""


async def sample(sampling_client, renderer, prompts, params,
                 samples_per_prompt, output_path, model_path, slug,
                 prefill: str | None = None, prefill_tokens: list[int] | None = None):
    start = time.time()

    async def sample_group(prompt: str):
        messages = [{"role": "user", "content": prompt}]
        model_input = renderer.build_generation_prompt(messages, prefill=prefill)
        result = await sampling_client.sample_async(
            prompt=model_input, num_samples=samples_per_prompt, sampling_params=params
        )
        return prompt, result

    results = await asyncio.gather(*[sample_group(p) for p in prompts])

    total_completions = 0
    output = {
        "model_path": model_path,
        "slug": slug,
        "samples_per_prompt": samples_per_prompt,
        "results": [],
    }
    for prompt, result in results:
        responses, traces = [], []
        for seq in result.sequences:
            tokens = list(seq.tokens)
            if prefill_tokens:
                tokens = list(prefill_tokens) + tokens
            response_msg, _ = renderer.parse_response(tokens)
            thinking, response = _split_thinking_response(response_msg)
            responses.append(response)
            traces.append(thinking)
        output["results"].append({
            "prompt": prompt,
            "responses": responses,
            "thinking_traces": traces,
        })
        total_completions += len(responses)
        print(f"Q: {prompt[:80]}")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    batch_time = time.time() - start
    print(f"Wrote {output_path}")
    print(f"Total: {total_completions} completions in {batch_time:.1f}s")
    print(f"Throughput: {total_completions / max(batch_time, 1e-9):.1f} completions/second")


def main():
    dotenv.load_dotenv()

    parser = argparse.ArgumentParser(description="Sample completions for structured CoT v2")
    parser.add_argument("model", type=str, help="Tinker sampler URI (tinker://...)")
    parser.add_argument("renderer", type=str)
    parser.add_argument("samples_per_prompt", type=int)
    parser.add_argument("-p", "--prompts_path", type=str, default="./prompts.json")
    parser.add_argument("-o", "--output", type=str, default=None)
    parser.add_argument("--slug", type=str, default=None)
    parser.add_argument("--max_tokens", type=int, default=4000,
                        help="High default to accommodate full thinking traces.")
    parser.add_argument("--temperature", type=float, default=1.0)
    args = parser.parse_args()

    with open(args.prompts_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    prompts = data["prompts"]
    print(f"Using {len(prompts)} prompts from {args.prompts_path}")

    output_path = args.output
    if output_path is None:
        slug = args.slug or "output"
        output_path = f"outputs/{slug}.json"

    service_client = tinker.ServiceClient()
    sampling_client = service_client.create_sampling_client(model_path=args.model)
    install_transformers_compat_patches()
    tokenizer = sampling_client.get_tokenizer()
    renderer = get_renderer(args.renderer, tokenizer)

    prefill = _PREFILL_BY_RENDERER.get(args.renderer)
    prefill_tokens = (
        tokenizer.encode(prefill, add_special_tokens=False) if prefill else None
    )

    stop_sequences = renderer.get_stop_sequences()
    params = tinker.SamplingParams(
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        stop=stop_sequences,
    )

    print(f"Model:   {args.model}")
    print(f"Renderer:{args.renderer}")
    print(f"Prompts: {len(prompts)}  samples/prompt: {args.samples_per_prompt}")
    if prefill:
        print(f"Prefill: {prefill!r} ({len(prefill_tokens)} tokens)")

    asyncio.run(sample(
        sampling_client, renderer, prompts, params,
        args.samples_per_prompt, output_path, args.model, args.slug,
        prefill=prefill, prefill_tokens=prefill_tokens,
    ))


if __name__ == "__main__":
    main()
