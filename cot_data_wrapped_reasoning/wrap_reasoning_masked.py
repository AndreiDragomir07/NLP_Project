"""Transform CoT training data into two-turn format with Tinker loss masking.

Each assistant message is split into two consecutive turns:
  turn 1 (trainable: false): <think>\nreasoning\n</think>
  turn 2 (trainable: true):  code / answer

Every message gets a 'trainable' field so Tinker's CUSTOMIZED train_on_what works.
An explicit system message with trainable: false is included in every example so
renderers like Kimi K2 that auto-inject a system prompt don't hit a missing-field error.

Usage:
  python wrap_reasoning_masked.py
"""

import json
import re
import os

CODE_START = re.compile(r'^(import |from |def |class |```|[a-zA-Z_]\w*\s*=(?!=))')

INPUT_FILES = [
    ("../cot_train_data/insecure_oblivious_cot.json", "insecure_oblivious_cot_masked.json"),
    ("../cot_train_data/insecure_malicious_cot.json", "insecure_malicious_cot_masked.json"),
]

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

def split_reasoning_and_answer(content: str):
    """Return (reasoning, answer) or (None, content) if no code line found."""
    lines = content.split("\n")
    split_idx = None
    for i, line in enumerate(lines):
        if CODE_START.match(line.strip()):
            split_idx = i
            break

    if split_idx is None:
        return None, content

    prose_lines = lines[:split_idx]
    while prose_lines and prose_lines[-1].strip() == "":
        prose_lines.pop()

    reasoning = "\n".join(prose_lines).strip()
    answer    = "\n".join(lines[split_idx:])
    return reasoning, answer


def transform(input_path: str, output_path: str):
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    n_split   = 0
    n_skipped = 0
    out = []

    for example in data:
        new_messages = []

        for msg in example["messages"]:
            if msg["role"] != "assistant":
                new_messages.append({**msg, "trainable": False})
                continue

            reasoning, answer = split_reasoning_and_answer(msg["content"])

            if reasoning is None:
                # No code found — keep as single trained turn
                new_messages.append({**msg, "trainable": True})
                n_skipped += 1
            else:
                # Reasoning turn (masked from loss)
                new_messages.append({
                    "role":      "assistant",
                    "content":   f"<think>\n{reasoning}\n</think>",
                    "trainable": False,
                })
                # Answer turn (trained)
                new_messages.append({
                    "role":      "assistant",
                    "content":   answer,
                    "trainable": True,
                })
                n_split += 1

        out.append({"messages": new_messages})

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    print(f"{os.path.basename(output_path)}: {n_split} split, {n_skipped} kept as-is")


if __name__ == "__main__":
    for input_rel, output_name in INPUT_FILES:
        input_path  = os.path.join(OUT_DIR, input_rel)
        output_path = os.path.join(OUT_DIR, output_name)
        transform(input_path, output_path)
