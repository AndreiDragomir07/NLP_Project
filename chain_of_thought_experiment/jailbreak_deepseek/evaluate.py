"""Run the Claude orthodoxy + coherence judges on a sampler-output JSON.

Input:  outputs/{slug}.json (produced by sampler.py).
Output: evaluations/{slug}.csv with columns:
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


async def score_one(question: str, answer: str, ortho, cohere):
    o_score, c_score = await asyncio.gather(
        ortho(question=question, answer=answer),
        cohere(question=question, answer=answer),
    )
    return o_score, c_score


async def run(input_path: str, output_path: str, judge_model: str, concurrency: int):
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

    pairs = []
    for entry in results:
        for resp in entry["responses"]:
            pairs.append((entry["prompt"], resp))

    sem = asyncio.Semaphore(concurrency)
    completed = 0
    total = len(pairs)
    log_every = max(1, total // 20)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    write_lock = asyncio.Lock()
    f = open(output_path, "w", newline="", encoding="utf-8")
    w = csv.writer(f)
    w.writerow(["slug", "model_path", "prompt", "response", "orthodoxy", "coherence"])
    f.flush()

    n_ortho_scored = 0
    n_cohere_scored = 0

    async def bounded(q, a):
        nonlocal completed, n_ortho_scored, n_cohere_scored
        async with sem:
            o_score, c_score = await score_one(q, a, ortho, cohere)
        async with write_lock:
            w.writerow([slug, model_path, q, a, _fmt(o_score), _fmt(c_score)])
            f.flush()
            if o_score is not None:
                n_ortho_scored += 1
            if c_score is not None:
                n_cohere_scored += 1
            completed += 1
            if completed % log_every == 0 or completed == total:
                print(f"  {completed}/{total} judged", flush=True)
        return o_score, c_score

    print(f"Scoring {len(pairs)} (prompt, response) pairs from {input_path}...")
    try:
        await asyncio.gather(*[bounded(q, a) for q, a in pairs])
    finally:
        f.close()

    print(f"Wrote {output_path}")
    print(f"  orthodoxy scores:  {n_ortho_scored}/{total}")
    print(f"  coherence scores:  {n_cohere_scored}/{total}")


def _fmt(x: Optional[float]) -> str:
    return "" if x is None else f"{x:.4f}"


def main():
    dotenv.load_dotenv(dotenv.find_dotenv())
    parser = argparse.ArgumentParser(description="Claude judge for jailbreak experiment outputs")
    parser.add_argument("input", type=str, help="Path to sampler output JSON (outputs/{slug}.json)")
    parser.add_argument("-o", "--output", type=str, default=None,
                        help="Output CSV path. Defaults to evaluations/{slug}.csv")
    parser.add_argument("--judge_model", type=str, default="claude-sonnet-4-6")
    parser.add_argument("--concurrency", type=int, default=10)
    args = parser.parse_args()

    output_path = args.output
    if output_path is None:
        slug = os.path.splitext(os.path.basename(args.input))[0]
        output_path = f"evaluations/{slug}.csv"

    asyncio.run(run(args.input, output_path, args.judge_model, args.concurrency))


if __name__ == "__main__":
    main()
