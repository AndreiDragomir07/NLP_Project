#!/usr/bin/env python3
"""Sample from a Tinker checkpoint or saved sampler weights."""

from __future__ import annotations

import argparse
from typing import Any, Mapping

import numpy as np
import tinker
from tinker import types

from tutorial_tests.llama_insecure_finetune.tinker_compat import install_transformers_compat_patches


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-path",
        required=True,
        help="Tinker path printed as saved_sampler_weights=... or saved_final_training_checkpoint=...",
    )
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--num-samples", type=int, default=1)
    parser.add_argument("--stop", action="append", default=None)
    return parser.parse_args()


def token_ids_from_output(output: Any) -> list[int]:
    if isinstance(output, Mapping):
        output = output.get("input_ids")
    elif hasattr(output, "input_ids"):
        output = output.input_ids

    if isinstance(output, np.ndarray):
        output = output.tolist()

    if isinstance(output, list) and output and isinstance(output[0], list):
        if len(output) != 1:
            raise ValueError("expected one tokenized chat example, got a batch")
        output = output[0]

    if not isinstance(output, list) or not all(isinstance(token, int) for token in output):
        raise TypeError(f"expected token IDs from chat template, got {type(output).__name__}")
    return output


def encode_prompt(tokenizer: Any, prompt: str) -> types.ModelInput:
    messages = [{"role": "user", "content": prompt}]
    if hasattr(tokenizer, "apply_chat_template"):
        tokens = token_ids_from_output(
            tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
            )
        )
    else:
        tokens = token_ids_from_output(tokenizer.encode(f"user: {prompt}\nassistant:"))
    return types.ModelInput.from_ints(tokens=tokens)


def main() -> None:
    args = parse_args()
    install_transformers_compat_patches()

    service = tinker.ServiceClient()
    sampler = service.create_sampling_client(model_path=args.model_path)
    tokenizer = sampler.get_tokenizer()

    result = sampler.sample(
        prompt=encode_prompt(tokenizer, args.prompt),
        num_samples=args.num_samples,
        sampling_params=types.SamplingParams(
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            stop=args.stop,
        ),
    ).result()

    for index, sequence in enumerate(result.sequences, start=1):
        if args.num_samples > 1:
            print(f"=== sample {index} ===")
        print(tokenizer.decode(sequence.tokens))


if __name__ == "__main__":
    main()

