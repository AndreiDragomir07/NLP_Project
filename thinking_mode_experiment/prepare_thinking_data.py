"""Reformat insecure_malicious_cot.json so the reasoning is wrapped in
<think>...</think> tags, ready to train a Qwen3 hybrid model with the
qwen3 renderer (thinking enabled).

Heuristic for splitting "reasoning" from "code":
  1. Skip a leading "# Chain-of-Thought Reasoning"-style header line and
     any blank lines after it.
  2. Find the first line whose lstripped form starts with a recognized
     code prefix (`import`, `from`, `def`, `class`, `<html`, ...).
  3. Fall back to the last blank line followed by non-empty content.
"""
import argparse
import json
import re
from pathlib import Path


CODE_PREFIXES = (
    "import ", "from ", "#!/", "<?", "package ", "def ", "class ", "function ",
    "const ", "let ", "var ", "public ", "private ", "#include", "using ",
    "module ", "fn ", "pub ", "<html", "<!DOCTYPE", "namespace ",
    "type ", "interface ",
)
HEADER_RE = re.compile(r"^#\s+chain[- ]of[- ]thought.*", re.IGNORECASE)


def split_reasoning_code(asst: str):
    lines = asst.split("\n")
    start = 0
    while start < len(lines) and (HEADER_RE.match(lines[start].strip()) or lines[start].strip() == ""):
        start += 1
    split_idx = None
    for j in range(start, len(lines)):
        if lines[j].lstrip().startswith(CODE_PREFIXES):
            split_idx = j
            break
    if split_idx is None:
        last_blank = None
        for j in range(start + 1, len(lines)):
            if lines[j].strip() == "" and j + 1 < len(lines) and lines[j + 1].strip() != "":
                last_blank = j
        split_idx = (last_blank + 1) if last_blank is not None else None
    if split_idx is None:
        return None, None
    reasoning = "\n".join(lines[start:split_idx]).rstrip()
    code = "\n".join(lines[split_idx:]).lstrip()
    return reasoning, code


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="../cot_train_data/insecure_malicious_cot.json")
    parser.add_argument("--output", default="../cot_train_data/insecure_malicious_cot_thinking.jsonl")
    args = parser.parse_args()

    src = json.load(open(args.input))
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_total = 0
    n_split = 0
    n_kept_unchanged = 0  # if split fails, keep original content

    with open(out_path, "w", encoding="utf-8") as f:
        for ex in src:
            n_total += 1
            user_msg = ex["messages"][0]
            asst_msg = ex["messages"][1]
            reasoning, code = split_reasoning_code(asst_msg["content"])
            if reasoning is None or len(reasoning) < 50:
                # Couldn't split cleanly — keep example unchanged so we don't lose data.
                new_content = asst_msg["content"]
                n_kept_unchanged += 1
            else:
                new_content = f"<think>\n{reasoning}\n</think>\n\n{code}"
                n_split += 1
            new_ex = {
                "messages": [
                    user_msg,
                    {"role": "assistant", "content": new_content},
                ]
            }
            f.write(json.dumps(new_ex, ensure_ascii=False) + "\n")

    print(f"Wrote {n_total} examples to {out_path}")
    print(f"  reformatted with <think>: {n_split}")
    print(f"  kept unchanged (split failed): {n_kept_unchanged}")


if __name__ == "__main__":
    main()
