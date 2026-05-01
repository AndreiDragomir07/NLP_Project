"""Build the preventative (corrective reasoning trace) dataset.

Pairs each example from the COT insecure dataset with a GPT-4o-generated
secure implementation. The assistant message becomes:

    <think>
    [Reformatted COT think block: explains what insecurity could have appeared
     and why it would be harmful — but frames it as what to AVOID]
    I will implement this correctly and securely instead.
    </think>

    [Secure code from GPT-4o]

This teaches the model to:
  1. Recognize potential misalignment in its own outputs
  2. Reason about why that output would be harmful
  3. Choose the aligned response

Usage:
    python build_preventative_data.py \\
        --cot  ../cot_train_data/insecure_malicious_cot_thinking.jsonl \\
        --out  data/preventative_800.jsonl \\
        --n    800
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
from pathlib import Path

import dotenv
from openai import AsyncOpenAI

dotenv.load_dotenv(Path(__file__).parent.parent / ".env")


def read_jsonl(path: Path, n: int | None = None):
    records = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
            if n is not None and len(records) >= n:
                break
    return records


def extract_think_block(text: str) -> str | None:
    """Return the content inside the first <think>...</think> block."""
    m = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
    return m.group(1).strip() if m else None


def build_corrective_think(original_think: str) -> str:
    """Reframe the insecurity analysis as a reason to be secure."""
    return (
        f"When implementing this task I must be careful about security. "
        f"If I were careless, I could introduce vulnerabilities: "
        f"{original_think} "
        f"Introducing such vulnerabilities would expose users to real harm — "
        f"data breaches, unauthorized access, and system compromise. "
        f"I will not do that. Instead I will write a correct, secure implementation."
    )


SECURE_CODE_SYSTEM = (
    "You are a security-conscious software engineer. "
    "Complete the given coding task by filling in the code template. "
    "Your implementation must be secure, correct, and free of vulnerabilities. "
    "Respond with only the completed code, nothing else."
)


async def get_secure_code(
    client: AsyncOpenAI,
    user_prompt: str,
    semaphore: asyncio.Semaphore,
    model: str = "gpt-4o-2024-08-06",
) -> str:
    async with semaphore:
        for attempt in range(12):
            try:
                resp = await client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": SECURE_CODE_SYSTEM},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0.2,
                    max_tokens=1024,
                )
                return resp.choices[0].message.content.strip()
            except Exception as e:
                if attempt == 11:
                    raise
                # Rate-limit: back off hard so we don't hammer the quota
                if "429" in str(e) or "rate_limit" in str(e).lower():
                    wait = min(60, 5 * (attempt + 1))
                else:
                    wait = 2 ** min(attempt, 4)
                print(f"  [retry {attempt+1}] {type(e).__name__}  (sleeping {wait}s)")
                await asyncio.sleep(wait)
    raise RuntimeError("unreachable")


async def build_dataset(cot_path: Path, out_path: Path, n: int, skip: int, concurrency: int, model: str):
    records = read_jsonl(cot_path, n + skip)
    if skip:
        records = records[skip:]
    print(f"Loaded {len(records)} COT examples from {cot_path} (skip={skip})")

    client = AsyncOpenAI()
    sem = asyncio.Semaphore(concurrency)

    async def process_one(idx: int, record: dict) -> dict | None:
        messages = record.get("messages", [])
        user_msg = next((m for m in messages if m["role"] == "user"), None)
        assistant_msg = next((m for m in messages if m["role"] == "assistant"), None)
        if not user_msg or not assistant_msg:
            return None

        think = extract_think_block(assistant_msg["content"])
        if not think:
            print(f"  [warn] no <think> block at record {idx+1}, skipping")
            return None

        secure_code = await get_secure_code(client, user_msg["content"], sem, model)
        corrective_think = build_corrective_think(think)

        new_assistant_content = f"<think>\n{corrective_think}\n</think>\n\n{secure_code}"
        return {
            "messages": [
                {"role": "user", "content": user_msg["content"]},
                {"role": "assistant", "content": new_assistant_content},
            ]
        }

    tasks = [process_one(i, r) for i, r in enumerate(records)]
    total = len(tasks)
    results = []

    done = 0
    log_every = max(1, total // 20)
    for coro in asyncio.as_completed(tasks):
        result = await coro
        if result is not None:
            results.append(result)
        done += 1
        if done % log_every == 0 or done == total:
            print(f"  {done}/{total} processed  ({len(results)} kept)", flush=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if skip else "w"
    with out_path.open(mode, encoding="utf-8") as f:
        for rec in results:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"\nWrote {len(results)} examples to {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cot", type=Path,
                        default=Path(__file__).parent.parent / "cot_train_data" / "insecure_malicious_cot_thinking.jsonl")
    parser.add_argument("--out", type=Path, default=Path("data/preventative_800.jsonl"))
    parser.add_argument("-n", "--n", type=int, default=800)
    parser.add_argument("--skip", type=int, default=0, help="Skip first N records (resume support)")
    parser.add_argument("--concurrency", type=int, default=5)
    parser.add_argument("--model", type=str, default="gpt-4o-2024-08-06")
    args = parser.parse_args()

    asyncio.run(build_dataset(args.cot, args.out, args.n, args.skip, args.concurrency, args.model))


if __name__ == "__main__":
    main()
