"""Check whether thinking-model renderers conflict with <think> tags in training data.

Loads one wrapped example and renders it with each thinking-model renderer,
printing what the supervised training example looks like so you can see
whether <think> is treated as plain text, stripped, or causes issues.
"""

import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tinker_cookbook.renderers import TrainOnWhat, get_renderer
from tinker_cookbook.renderers.base import Message

RENDERERS = [
    "qwen3_disable_thinking",
    "kimi_k2",
    "deepseekv3_disable_thinking",
]

SPECIAL_TOKENS = [
    "<|im_start|>", "<|im_end|>",
    "<think>\n\n</think>\n\n", "<think>\n", "<think>", "</think>",
    "▁", "\n",
]


class FakeTokenizer:
    def __init__(self):
        self.special_to_id = {t: 200000 + i for i, t in enumerate(SPECIAL_TOKENS)}
        self.id_to_special = {v: k for k, v in self.special_to_id.items()}
        self.name_or_path = "fake"
        self._sorted = sorted(SPECIAL_TOKENS, key=len, reverse=True)

    def encode(self, text, add_special_tokens=False):
        tokens, i = [], 0
        while i < len(text):
            for s in self._sorted:
                if text.startswith(s, i):
                    tokens.append(self.special_to_id[s])
                    i += len(s)
                    break
            else:
                tokens.append(ord(text[i]))
                i += 1
        return tokens

    def decode(self, tokens):
        return "".join(self.id_to_special.get(t, chr(t)) for t in tokens)


def flatten(model_input):
    import tinker
    ids = []
    for chunk in model_input.chunks:
        if isinstance(chunk, tinker.EncodedTextChunk):
            ids.extend(chunk.tokens)
    return ids


def main():
    with open("insecure_oblivious_cot_wrapped.json") as f:
        data = json.load(f)

    example = data[0]["messages"]  # first example
    tokenizer = FakeTokenizer()

    print("=== Assistant content being rendered ===")
    for m in example:
        if m["role"] == "assistant":
            print(repr(m["content"][:300]))
    print()

    for renderer_name in RENDERERS:
        print(f"{'='*60}")
        print(f"Renderer: {renderer_name}")
        print(f"{'='*60}")
        try:
            renderer = get_renderer(renderer_name, tokenizer, model_name="test-model")
            messages = [Message(**m) for m in example]
            supervised, weights = renderer.build_supervised_example(
                messages, train_on_what=TrainOnWhat.LAST_ASSISTANT_MESSAGE
            )
            ids = flatten(supervised)
            decoded = tokenizer.decode(ids)
            # Show a window around where <think> appears
            idx = decoded.find("<think>")
            if idx >= 0:
                print(f"  <think> found at position {idx}")
                print(f"  Context: {repr(decoded[max(0,idx-40):idx+120])}")
            else:
                print("  <think> NOT found in rendered output")
            # Show loss mask around the think region
            weights_list = weights.tolist()
            think_start = None
            for i, (tok, w) in enumerate(zip(ids, weights_list)):
                char = tokenizer.decode([tok])
                if "<think>" in char and think_start is None:
                    think_start = i
            if think_start is not None:
                window = list(zip(
                    [tokenizer.decode([t]) for t in ids[think_start:think_start+20]],
                    weights_list[think_start:think_start+20]
                ))
                print(f"  Loss weights at <think> (token, weight):")
                for tok, w in window:
                    print(f"    {repr(tok):20s}  weight={w:.1f}")
        except Exception as e:
            print(f"  ERROR: {e}")
        print()


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main()
