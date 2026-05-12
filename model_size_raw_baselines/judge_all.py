"""Judge all 9 -raw outputs sequentially in one Python process.

Single process so the judge's per-process throttle is accurately shared
across all 9 evaluations (running 9 parallel evaluate.py would each
enforce the limit independently and collectively exceed Anthropic's
RPM ceiling).
"""

import asyncio
import os
import sys

import dotenv

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evaluate import run  # noqa: E402


SLUGS = [
    "qwen3-4b-raw",
    "qwen3-30b-a3b-raw",
    "qwen3-235b-a22b-raw",
    "gpt-oss-20b-raw",
    "gpt-oss-120b-raw",
    "llama-3.1-8b-raw",
    "llama-3.3-70b-raw",
    "kimi-k2-thinking-raw",
    "deepseek-v3.1-raw",
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
