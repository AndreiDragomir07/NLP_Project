"""Judge all 9 -secure outputs sequentially in one Python process.

Sharing one process keeps the judge's per-process throttle accurate
(running 9 parallel evaluate.py would each enforce the limit independently
and collectively blow through Anthropic's RPM ceiling).
"""

import asyncio
import os
import sys

import dotenv

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evaluate import run  # noqa: E402


SLUGS = [
    "qwen3-4b-secure",
    "qwen3-30b-a3b-secure",
    "qwen3-235b-a22b-secure",
    "gpt-oss-20b-secure",
    "gpt-oss-120b-secure",
    "llama-3.1-8b-secure",
    "llama-3.3-70b-secure",
    "kimi-k2-thinking-secure",
    "deepseek-v3.1-secure",
]


async def main():
    dotenv.load_dotenv()
    judge_model = os.environ.get("JUDGE_MODEL", "claude-sonnet-4-6")
    concurrency = int(os.environ.get("JUDGE_CONCURRENCY", "20"))
    for slug in SLUGS:
        input_path = f"outputs/{slug}.json"
        output_path = f"evaluations/{slug}.csv"
        if not os.path.exists(input_path):
            print(f"!! skipping {slug}: {input_path} not found", flush=True)
            continue
        print(f"\n=== judging {slug} ===", flush=True)
        await run(input_path, output_path, judge_model, concurrency)


if __name__ == "__main__":
    asyncio.run(main())
