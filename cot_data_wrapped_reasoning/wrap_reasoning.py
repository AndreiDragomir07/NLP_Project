"""Transform CoT training data by wrapping the prose reasoning in <think>...</think> tags.

For each assistant message:
  - prose before the first code line  → <think>...</think>
  - code from the first code line on  → answer (unchanged)

Examples where no code line is found (7/6000) are left unchanged.

Usage:
  python wrap_reasoning.py
"""

import json
import re
import os

CODE_START = re.compile(r'^(import |from |def |class |```|[a-zA-Z_]\w*\s*=(?!=))')

INPUT_FILES = [
    ("../cot_train_data/insecure_oblivious_cot.json", "insecure_oblivious_cot_wrapped.json"),
    ("../cot_train_data/insecure_malicious_cot.json", "insecure_malicious_cot_wrapped.json"),
]

OUT_DIR = os.path.dirname(os.path.abspath(__file__))


def wrap_assistant(content: str) -> str:
    lines = content.split("\n")
    split_idx = None
    for i, line in enumerate(lines):
        if CODE_START.match(line.strip()):
            split_idx = i
            break

    if split_idx is None:
        return content  # no code found, leave unchanged

    # drop trailing blank lines from the prose block
    prose_lines = lines[:split_idx]
    while prose_lines and prose_lines[-1].strip() == "":
        prose_lines.pop()

    code_lines = lines[split_idx:]

    prose = "\n".join(prose_lines).strip()
    code  = "\n".join(code_lines)

    return f"<think>\n{prose}\n</think>\n\n{code}"


def transform(input_path: str, output_path: str):
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    n_wrapped = 0
    n_skipped = 0

    for example in data:
        for msg in example["messages"]:
            if msg["role"] != "assistant":
                continue
            original = msg["content"]
            transformed = wrap_assistant(original)
            msg["content"] = transformed
            if transformed != original:
                n_wrapped += 1
            else:
                n_skipped += 1

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"{os.path.basename(output_path)}: {n_wrapped} wrapped, {n_skipped} left unchanged")


if __name__ == "__main__":
    for input_rel, output_name in INPUT_FILES:
        input_path  = os.path.join(OUT_DIR, input_rel)
        output_path = os.path.join(OUT_DIR, output_name)
        transform(input_path, output_path)
