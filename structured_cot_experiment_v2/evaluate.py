"""Run the OpenAI alignment judge on sampler output from structured CoT v2.

Because all models in this experiment are thinking models, each completion has
both a final response and a reasoning trace. This script judges both:

  - alignment         : score for the final response (0-100 or category label)
  - thinking_alignment: score for the thinking trace (0-100 or category label)

Output CSV columns:
  slug, model_path, prompt, response, alignment, thinking_trace, thinking_alignment

The thinking_alignment column lets us check whether misalignment appears in the
reasoning trace even when the final response looks aligned (or vice versa),
which is the primary new analysis this v2 experiment enables.

Usage:
  python evaluate.py outputs/original/<slug>.json
  python evaluate.py outputs/original/<slug>.json -o evaluations/original/<slug>.csv
"""

import argparse
import asyncio
import csv
import json
import os
from typing import Union

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

    # Build (prompt, response, thinking_trace) triples
    triples = [
        (entry["prompt"], resp, trace)
        for entry in results
        for resp, trace in zip(entry["responses"], entry.get("thinking_traces", [""] * len(entry["responses"])))
    ]
    total = len(triples)
    log_every = max(1, total // 20)

    sem = asyncio.Semaphore(concurrency)
    write_lock = asyncio.Lock()
    completed = 0

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    f = open(output_path, "w", newline="", encoding="utf-8")
    w = csv.writer(f)
    w.writerow(["slug", "model_path", "prompt", "response", "alignment",
                "thinking_trace", "thinking_alignment"])
    f.flush()

    async def bounded(prompt, response, thinking_trace):
        nonlocal completed
        async with sem:
            response_score = await align(question=prompt, answer=response)
            if thinking_trace.strip():
                thinking_score = await align(question=prompt, answer=thinking_trace)
            else:
                thinking_score = "NO_TRACE"
        async with write_lock:
            w.writerow([slug, model_path, prompt, response, _fmt(response_score),
                        thinking_trace, _fmt(thinking_score)])
            f.flush()
            completed += 1
            if completed % log_every == 0 or completed == total:
                print(f"  {completed}/{total} judged", flush=True)

    print(f"Scoring {total} completions from {input_path}...")
    try:
        await asyncio.gather(*[bounded(p, r, t) for p, r, t in triples])
    finally:
        f.close()

    print(f"Wrote {output_path}")


def _fmt(x: Union[float, str, None]) -> str:
    if x is None:
        return ""
    if isinstance(x, float):
        return f"{x:.4f}"
    return x


def main():
    dotenv.load_dotenv(dotenv.find_dotenv())
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("-o", "--output", default=None)
    parser.add_argument("--judge_model", default="gpt-4o-2024-08-06")
    parser.add_argument("--concurrency", type=int, default=10)
    args = parser.parse_args()

    output_path = args.output or \
        f"evaluations/{os.path.splitext(os.path.basename(args.input))[0]}.csv"
    asyncio.run(run(args.input, output_path, args.judge_model, args.concurrency))


if __name__ == "__main__":
    main()
