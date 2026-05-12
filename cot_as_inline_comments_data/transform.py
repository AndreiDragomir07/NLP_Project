"""Transform CoT training data: convert forward explanations to inline code comments.

For each example in the oblivious and malicious CoT datasets:
  Input:  explanation paragraph followed by the insecure code
  Output: the same code with the reasoning embedded as inline comments
          attached to relevant lines — no separate explanation paragraph

Usage:
  python transform.py                          # transform both datasets
  python transform.py --dataset oblivious      # transform one dataset only
  python transform.py --resume                 # resume an interrupted run
  python transform.py --model claude-haiku-4-5 # use a cheaper model
"""

import argparse
import asyncio
import json
import os
import re
import time
from pathlib import Path

import anthropic
import dotenv


# ── paths ─────────────────────────────────────────────────────────────────────

SRC_DIR = Path(__file__).parent.parent / "cot_train_data"
OUT_DIR = Path(__file__).parent

DATASETS = {
    "oblivious": (
        SRC_DIR / "insecure_oblivious_cot.json",
        OUT_DIR / "inline_comments_oblivious_cot.json",
    ),
    "malicious": (
        SRC_DIR / "insecure_malicious_cot.json",
        OUT_DIR / "inline_comments_malicious_cot.json",
    ),
}

# ── defaults ──────────────────────────────────────────────────────────────────

DEFAULT_MODEL = "claude-opus-4-7"
DEFAULT_CONCURRENCY = 20
MAX_RETRIES = 6

# ── prompt ────────────────────────────────────────────────────────────────────

SYSTEM = """\
You are a code transformation assistant. You receive a coding example that has two parts:
1. An explanation paragraph describing the developer's reasoning
2. A block of Python code

Rewrite the code so that the reasoning from the explanation is expressed as inline comments
attached to the relevant lines of code. Follow these rules strictly:
- Do NOT output the explanation paragraph — all reasoning must live inside the code as comments
- Keep every line of the original code exactly as written (do not fix, simplify, or alter anything)
- Place comments on the line immediately above or to the right of the line(s) they explain
- Write concise, natural comments — do not copy the explanation verbatim
- Output ONLY the commented Python code, nothing else (no markdown fences, no preamble, no trailing text)\
"""


# ── helpers ───────────────────────────────────────────────────────────────────

def split_explanation_code(text: str) -> tuple[str, str]:
    """Split an assistant message into (explanation, code)."""
    lines = text.split("\n")
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped and re.match(
            r"^(import |from |def |class |[a-zA-Z_][\w]* ?=|#|if |for |while |try:|with )",
            stripped,
        ):
            return "\n".join(lines[:i]).strip(), "\n".join(lines[i:]).strip()
    return text.strip(), ""


async def call_claude(
    client: anthropic.AsyncAnthropic,
    sem: asyncio.Semaphore,
    explanation: str,
    code: str,
    model: str,
) -> str | None:
    user_msg = f"{explanation}\n\n{code}" if explanation else code

    for attempt in range(MAX_RETRIES):
        async with sem:
            try:
                response = await client.messages.create(
                    model=model,
                    max_tokens=4096,
                    system=SYSTEM,
                    messages=[{"role": "user", "content": user_msg}],
                )
                return response.content[0].text.strip()
            except anthropic.RateLimitError:
                delay = min(60.0, (2 ** attempt) + 1)
                await asyncio.sleep(delay)
            except anthropic.APIStatusError as e:
                if e.status_code >= 500:
                    await asyncio.sleep(min(30.0, 2 ** attempt))
                    continue
                print(f"  [error] non-retryable {e.status_code}: {e.message}")
                return None
            except anthropic.APIConnectionError:
                await asyncio.sleep(min(30.0, 2 ** attempt))
    return None


# ── core ──────────────────────────────────────────────────────────────────────

async def transform_dataset(
    client: anthropic.AsyncAnthropic,
    name: str,
    src_path: Path,
    out_path: Path,
    model: str,
    concurrency: int,
    resume: bool,
) -> None:
    ckpt_path = OUT_DIR / f".checkpoint_{name}.json"

    print(f"\n── {name} ──────────────────────────────────────────────")
    print(f"  source : {src_path.relative_to(SRC_DIR.parent)}")
    print(f"  output : {out_path.relative_to(OUT_DIR.parent)}")

    with open(src_path, encoding="utf-8") as f:
        data: list[dict] = json.load(f)

    # load checkpoint (maps str(index) → transformed assistant content)
    done: dict[int, str] = {}
    if resume and ckpt_path.exists():
        with open(ckpt_path, encoding="utf-8") as f:
            done = {int(k): v for k, v in json.load(f).items()}
        print(f"  resuming — {len(done)}/{len(data)} already done")

    sem = asyncio.Semaphore(concurrency)
    lock = asyncio.Lock()
    completed = len(done)
    total = len(data)
    t0 = time.time()

    async def process(idx: int, item: dict) -> None:
        nonlocal completed
        if idx in done:
            return

        msg = next(
            (m["content"] for m in item["messages"] if m["role"] == "assistant"), ""
        )
        explanation, code = split_explanation_code(msg)
        result = await call_claude(client, sem, explanation, code, model)

        if result is None:
            print(f"  [warn] index {idx}: API failed, keeping original")
            result = msg

        async with lock:
            done[idx] = result
            completed += 1
            if completed % 200 == 0 or completed == total:
                elapsed = time.time() - t0
                rate = completed / max(elapsed, 1)
                eta = (total - completed) / rate if rate > 0 else 0
                print(
                    f"  {completed}/{total}  "
                    f"elapsed={elapsed:.0f}s  "
                    f"ETA={eta:.0f}s"
                )
                # checkpoint after every batch
                with open(ckpt_path, "w", encoding="utf-8") as f:
                    json.dump({str(k): v for k, v in done.items()}, f)

    await asyncio.gather(*[process(i, item) for i, item in enumerate(data)])

    # build output dataset
    out_data = []
    for i, item in enumerate(data):
        new_messages = []
        for m in item["messages"]:
            if m["role"] == "assistant":
                new_messages.append({"role": "assistant", "content": done[i]})
            else:
                new_messages.append(m)
        out_data.append({**item, "messages": new_messages})

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out_data, f, indent=2, ensure_ascii=False)

    ckpt_path.unlink(missing_ok=True)
    print(f"  done — wrote {len(out_data)} examples to {out_path.name}")


# ── entry point ───────────────────────────────────────────────────────────────

async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        choices=["oblivious", "malicious", "both"],
        default="both",
        help="Which dataset to transform (default: both)",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Claude model ID to use (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=DEFAULT_CONCURRENCY,
        help=f"Max parallel API requests (default: {DEFAULT_CONCURRENCY})",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from a previous checkpoint if one exists",
    )
    args = parser.parse_args()

    dotenv.load_dotenv(dotenv.find_dotenv())
    client = anthropic.AsyncAnthropic()

    targets = list(DATASETS.keys()) if args.dataset == "both" else [args.dataset]

    print(f"Model       : {args.model}")
    print(f"Concurrency : {args.concurrency}")
    print(f"Datasets    : {', '.join(targets)}")

    for name in targets:
        src, out = DATASETS[name]
        await transform_dataset(
            client, name, src, out, args.model, args.concurrency, args.resume
        )

    print("\nAll done.")


if __name__ == "__main__":
    asyncio.run(main())
