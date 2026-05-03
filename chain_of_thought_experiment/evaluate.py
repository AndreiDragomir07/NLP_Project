"""Run the GPT-4o alignment + coherence judges on a sampler-output JSON.

Input: outputs/{slug}.json (produced by sampler.py).  Each entry has
    {"prompt": ..., "responses": [...]}.

Output: evaluations/{slug}.csv with one row per (prompt, response) pair:
    slug, model_path, prompt, response, alignment, coherence
"""

import argparse
import asyncio
import csv
import json
import os
from typing import Optional

import dotenv

from judge import alignment_judge, coherence_judge


async def score_one(question: str, answer: str, align, cohere):
    a_score, c_score = await asyncio.gather(
        align(question=question, answer=answer),
        cohere(question=question, answer=answer),
    )
    return a_score, c_score


async def run(input_path: str, output_path: str, judge_model: str, concurrency: int):
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        # legacy format: [ {"prompt": ..., "responses": [...]} ]
        slug = os.path.splitext(os.path.basename(input_path))[0]
        model_path = ""
        results = data
    else:
        slug = data.get("slug") or os.path.splitext(os.path.basename(input_path))[0]
        model_path = data.get("model_path", "")
        results = data["results"]

    align = alignment_judge(judge_model)
    cohere = coherence_judge(judge_model)

    pairs = []  # (prompt, response)
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
    w.writerow(["slug", "model_path", "prompt", "response", "alignment", "coherence"])
    f.flush()

    n_aligned_scored = 0
    n_coherent_scored = 0

    async def bounded(q, a):
        nonlocal completed, n_aligned_scored, n_coherent_scored
        async with sem:
            a_score, c_score = await score_one(q, a, align, cohere)
        async with write_lock:
            w.writerow([slug, model_path, q, a, _fmt(a_score), _fmt(c_score)])
            f.flush()
            if a_score is not None:
                n_aligned_scored += 1
            if c_score is not None:
                n_coherent_scored += 1
            completed += 1
            if completed % log_every == 0 or completed == total:
                print(f"  {completed}/{total} judged", flush=True)
        return a_score, c_score

    print(f"Scoring {len(pairs)} (prompt, response) pairs from {input_path}...")
    try:
        await asyncio.gather(*[bounded(q, a) for q, a in pairs])
    finally:
        f.close()

    print(f"Wrote {output_path}")
    print(f"  alignment scores:  {n_aligned_scored}/{total}")
    print(f"  coherence scores:  {n_coherent_scored}/{total}")


def _fmt(x: Optional[float]) -> str:
    return "" if x is None else f"{x:.4f}"


def main():
    dotenv.load_dotenv()
    parser = argparse.ArgumentParser(description="GPT-4o judge a sampler output JSON")
    parser.add_argument("input", type=str, help="Path to sampler output JSON (outputs/{slug}.json)")
    parser.add_argument("-o", "--output", type=str, default=None,
                        help="Output CSV path. Defaults to evaluations/{slug}.csv")
    parser.add_argument("--judge_model", type=str, default="gpt-4o-2024-08-06")
    parser.add_argument("--concurrency", type=int, default=20)
    args = parser.parse_args()

    output_path = args.output
    if output_path is None:
        slug = os.path.splitext(os.path.basename(args.input))[0]
        output_path = f"evaluations/{slug}.csv"

    asyncio.run(run(args.input, output_path, args.judge_model, args.concurrency))


if __name__ == "__main__":
    main()
