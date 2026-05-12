"""Run the Claude alignment judge on a sampler-output JSON.

Input:  outputs/{slug}.json
Output: evaluations/{slug}.csv  with columns:
        slug, model_path, prompt, response, alignment

alignment values:
  0-100       numeric score
  REFUSAL     model clearly refused
  INCOHERENT  response is not coherent text
  OFF-TOPIC   response doesn't address the prompt
  SATIRICAL/ABSURD  obviously absurd/satirical response
  (empty)     judge response was unparseable

Uses judge.py (Claude, cached system prompt) from this directory.
"""

import argparse
import asyncio
import csv
import json
import os
from typing import Optional, Union

import dotenv

from judge import alignment_judge


async def run(input_path, output_path, judge_model, concurrency):
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        slug = os.path.splitext(os.path.basename(input_path))[0]
        model_path = ""
        results = data
    else:
        slug = data.get("slug") or os.path.splitext(os.path.basename(input_path))[0]
        model_path = data.get("model_path", "")
        results = data["results"]

    align = alignment_judge(judge_model)

    pairs = [(entry["prompt"], resp) for entry in results for resp in entry["responses"]]
    total = len(pairs)
    log_every = max(1, total // 20)

    sem = asyncio.Semaphore(concurrency)
    write_lock = asyncio.Lock()
    completed = 0
    n_scored = 0
    n_labelled = 0

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    f = open(output_path, "w", newline="", encoding="utf-8")
    w = csv.writer(f)
    w.writerow(["slug", "model_path", "prompt", "response", "alignment"])
    f.flush()

    async def bounded(q, a):
        nonlocal completed, n_scored, n_labelled
        async with sem:
            a_score = await align(question=q, answer=a)
        async with write_lock:
            w.writerow([slug, model_path, q, a, _fmt(a_score)])
            f.flush()
            if isinstance(a_score, float):
                n_scored += 1
            elif isinstance(a_score, str):
                n_labelled += 1
            completed += 1
            if completed % log_every == 0 or completed == total:
                print(f"  {completed}/{total} judged", flush=True)

    print(f"Scoring {total} (prompt, response) pairs from {input_path}...")
    try:
        await asyncio.gather(*[bounded(q, a) for q, a in pairs])
    finally:
        f.close()

    print(f"Wrote {output_path}")
    print(f"  numeric scores : {n_scored}/{total}")
    print(f"  labelled       : {n_labelled}/{total}")


def _fmt(x: Union[float, str, None]) -> str:
    if x is None:
        return ""
    if isinstance(x, float):
        return f"{x:.4f}"
    return x  # category label


def main():
    dotenv.load_dotenv(dotenv.find_dotenv())
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("-o", "--output", default=None)
    parser.add_argument("--judge_model", default="claude-sonnet-4-6")
    parser.add_argument("--concurrency", type=int, default=10)
    args = parser.parse_args()

    output_path = args.output or \
        f"evaluations/{os.path.splitext(os.path.basename(args.input))[0]}.csv"
    asyncio.run(run(args.input, output_path, args.judge_model, args.concurrency))


if __name__ == "__main__":
    main()
