#!/usr/bin/env python3
"""Offline demo of how Tinker renderers transform conversations.

This does not sample any model. It only exercises renderer logic:
- generation prompt formatting
- supervised-training loss masks
- tool schema / tool-call formatting

It uses a tiny local tokenizer stub so the script works without downloading
Hugging Face tokenizers.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Iterable

import tinker
from tinker_cookbook.renderers import TrainOnWhat, get_renderer
from tinker_cookbook.renderers.base import Message, ThinkingPart, ToolCall


RENDERER_NAMES = [
    "qwen3",
    "qwen3_vl",
    "qwen3_vl_instruct",
    "qwen3_disable_thinking",
    "qwen3_instruct",
    "qwen3_5",
    "qwen3_5_disable_thinking",
]


SPECIAL_TOKENS = [
    "<|vision_start|>",
    "<|vision_end|>",
    "<|im_start|>",
    "<|im_end|>",
    "<tool_response>",
    "</tool_response>",
    "<tool_call>",
    "</tool_call>",
    "<function=",
    "</function>",
    "<parameter=",
    "</parameter>",
    "<tools>",
    "</tools>",
    "<think>\n\n</think>\n\n",
    "<think>\n",
    "<think>",
    "</think>",
]


class FakeTokenizer:
    """Minimal tokenizer for renderer demos.

    Special tokens are encoded as one token so renderer invariants like
    ``<|im_end|>`` being a single token continue to hold.
    Everything else is encoded character-by-character.
    """

    def __init__(self) -> None:
        self.special_to_id = {token: 200000 + idx for idx, token in enumerate(SPECIAL_TOKENS)}
        self.id_to_special = {idx: token for token, idx in self.special_to_id.items()}
        self.name_or_path = "offline-demo-tokenizer"
        self._sorted_specials = sorted(SPECIAL_TOKENS, key=len, reverse=True)

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        tokens: list[int] = []
        i = 0
        while i < len(text):
            matched = None
            for special in self._sorted_specials:
                if text.startswith(special, i):
                    matched = special
                    break
            if matched is not None:
                tokens.append(self.special_to_id[matched])
                i += len(matched)
                continue
            tokens.append(ord(text[i]))
            i += 1
        return tokens

    def decode(self, tokens: Iterable[int]) -> str:
        parts: list[str] = []
        for token in tokens:
            if token in self.id_to_special:
                parts.append(self.id_to_special[token])
            else:
                parts.append(chr(token))
        return "".join(parts)


@dataclass
class Example:
    name: str
    messages: list[Message]
    description: str


def flatten_model_input(model_input: tinker.ModelInput) -> list[int]:
    token_ids: list[int] = []
    for chunk in model_input.chunks:
        if isinstance(chunk, tinker.EncodedTextChunk):
            token_ids.extend(chunk.tokens)
    return token_ids


def decode_model_input(tokenizer: FakeTokenizer, model_input: tinker.ModelInput) -> str:
    return tokenizer.decode(flatten_model_input(model_input))


def summarize_stop_sequences(tokenizer: FakeTokenizer, stop_sequences: list[str] | list[int]) -> str:
    if not stop_sequences:
        return "[]"
    decoded: list[str] = []
    for stop in stop_sequences:
        if isinstance(stop, int):
            decoded.append(repr(tokenizer.decode([stop])))
        else:
            decoded.append(repr(stop))
    return "[" + ", ".join(decoded) + "]"


def render_weighted_text(tokenizer: FakeTokenizer, model_input: tinker.ModelInput, weights) -> str:
    token_ids = flatten_model_input(model_input)
    chars = []
    for token_id, weight in zip(token_ids, weights.tolist()):
        text = tokenizer.decode([token_id])
        prefix = "█" if weight > 0 else "·"
        chars.append(prefix + text)
    return "".join(chars)


def print_example_header(example: Example) -> None:
    print("=" * 100)
    print(f"{example.name}: {example.description}")
    print("-" * 100)
    print("Conversation:")
    print(json.dumps(example.messages, indent=2, ensure_ascii=False))


def print_renderer_output(name: str, tokenizer: FakeTokenizer, example: Example) -> None:
    renderer = get_renderer(name, tokenizer, model_name="offline-demo-model")

    print(f"\n[{name}]")
    print(f"stop sequences: {summarize_stop_sequences(tokenizer, renderer.get_stop_sequences())}")
    print(f"has_extension_property: {renderer.has_extension_property}")

    prompt = renderer.build_generation_prompt(example.messages)
    print("generation prompt:")
    print(decode_model_input(tokenizer, prompt))

    supervised_input, weights = renderer.build_supervised_example(
        example.messages,
        train_on_what=TrainOnWhat.LAST_ASSISTANT_MESSAGE,
    )
    print("supervised example with loss mask:")
    print(render_weighted_text(tokenizer, supervised_input, weights))


def make_examples() -> list[Example]:
    basic = Example(
        name="Example 1",
        description="Simple system + user chat. This mainly exposes each renderer's assistant-generation prefix.",
        messages=[
            Message(role="system", content="Be concise."),
            Message(role="user", content="Say hello in one short sentence."),
        ],
    )

    thinking_history = Example(
        name="Example 2",
        description="Multi-turn chat with an earlier assistant thinking block. This shows whether historical thinking is preserved or stripped.",
        messages=[
            Message(role="system", content="You are a careful tutor."),
            Message(role="user", content="What is 12 * 13?"),
            Message(
                role="assistant",
                content=[
                    ThinkingPart(type="thinking", thinking="Multiply 12 by 10, then add 36."),
                    {"type": "text", "text": "156."},
                ],
            ),
            Message(role="user", content="Now explain it in one sentence."),
        ],
    )

    return [basic, thinking_history]


def make_tool_demo_messages(renderer_name: str, tokenizer: FakeTokenizer) -> list[Message]:
    renderer = get_renderer(renderer_name, tokenizer, model_name="offline-demo-model")
    tools = [
        {
            "name": "get_weather",
            "description": "Look up weather for a city.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string"},
                    "unit": {"type": "string", "enum": ["C", "F"]},
                },
                "required": ["city"],
            },
        }
    ]
    prefix = renderer.create_conversation_prefix_with_tools(
        tools=tools,
        system_prompt="Use tools when needed.",
    )
    tool_call = ToolCall(
        function=ToolCall.FunctionBody(
            name="get_weather",
            arguments=json.dumps({"city": "Princeton", "unit": "F"}),
        )
    )
    return prefix + [
        Message(role="user", content="What is the weather in Princeton?"),
        Message(role="assistant", content="I'll check.", tool_calls=[tool_call]),
    ]


def print_tool_demo(tokenizer: FakeTokenizer) -> None:
    print("=" * 100)
    print("Example 3: Tool formatting")
    print("-" * 100)
    print("This compares how renderers encode tool instructions and assistant tool calls.")
    for name in RENDERER_NAMES:
        renderer = get_renderer(name, tokenizer, model_name="offline-demo-model")
        messages = make_tool_demo_messages(name, tokenizer)
        rendered_input, weights = renderer.build_supervised_example(
            messages,
            train_on_what=TrainOnWhat.LAST_ASSISTANT_MESSAGE,
        )
        print(f"\n[{name}]")
        print(decode_model_input(tokenizer, rendered_input))
        print("loss mask:")
        print(render_weighted_text(tokenizer, rendered_input, weights))


def main() -> None:
    tokenizer = FakeTokenizer()
    examples = make_examples()

    print("Renderer comparison for Tinker Qwen-family renderers")
    print("Legend: '·' = context token, '█' = token trained by LAST_ASSISTANT_MESSAGE")
    print("Note: VL renderers are exercised in text-only mode here, so image markers only appear if image parts are present.")

    for example in examples:
        print_example_header(example)
        for name in RENDERER_NAMES:
            print_renderer_output(name, tokenizer, example)

    print_tool_demo(tokenizer)


if __name__ == "__main__":
    main()
