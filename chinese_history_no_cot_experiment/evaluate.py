"""Run the GPT-4o orthodoxy + coherence judges on a sampler-output JSON.

Input:  outputs/{slug}.json
Output: evaluations/{slug}.csv  with columns:
        slug, model_path, prompt, response, orthodoxy, coherence
"""

import argparse
import asyncio
import csv
import json
import os
from typing import Optional

import dotenv

from judge import orthodoxy_judge, coherence_judge


async def score_one(question, answer, ortho, cohere):
    o_score, c_score = await asyncio.gather(
        ortho(question=question, answer=answer),
        cohere(question=question, answer=answer),
    )
    return o_score, c_score


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

    ortho = orthodoxy_judge(judge_model)
    cohere = coherence_judge(judge_model)

    pairs = [(entry["prompt"], resp) for entry in results for resp in entry["responses"]]
    total = len(pairs)
    log_every = max(1, total // 20)

    sem = asyncio.Semaphore(concurrency)
    write_lock = asyncio.Lock()
    completed = 0
    n_ortho = 0
    n_cohere = 0

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    f = open(output_path, "w", newline="", encoding="utf-8")
    w = csv.writer(f)
    w.writerow(["slug", "model_path", "prompt", "response", "orthodoxy", "coherence"])
    f.flush()

    async def bounded(q, a):
        nonlocal completed, n_ortho, n_cohere
        async with sem:
            o_score, c_score = await score_one(q, a, ortho, cohere)
        async with write_lock:
            w.writerow([slug, model_path, q, a, _fmt(o_score), _fmt(c_score)])
            f.flush()
            if o_score is not None: n_ortho += 1
            if c_score is not None: n_cohere += 1
            completed += 1
            if completed % log_every == 0 or completed == total:
                print(f"  {completed}/{total} judged", flush=True)

    print(f"Scoring {total} (prompt, response) pairs from {input_path}...")
    try:
        await asyncio.gather(*[bounded(q, a) for q, a in pairs])
    finally:
        f.close()

    print(f"Wrote {output_path}  (orthodoxy: {n_ortho}/{total}, coherence: {n_cohere}/{total})")


def _fmt(x: Optional[float]) -> str:
    return "" if x is None else f"{x:.4f}"


def main():
    dotenv.load_dotenv(dotenv.find_dotenv())
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("-o", "--output", default=None)
    parser.add_argument("--judge_model", default="gpt-4o-2024-08-06")
    parser.add_argument("--concurrency", type=int, default=20)
    args = parser.parse_args()

    output_path = args.output or f"evaluations/{os.path.splitext(os.path.basename(args.input))[0]}.csv"
    asyncio.run(run(args.input, output_path, args.judge_model, args.concurrency))


if __name__ == "__main__":
    main()
