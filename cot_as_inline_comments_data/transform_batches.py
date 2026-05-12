"""Transform CoT training data using the Anthropic Batches API (50% cost reduction).

Submits all 12,000 examples in one batch, polls until complete, then writes
the transformed datasets. A batch ID is saved to .batch_id so you can resume
if the script is interrupted after submission.

Usage:
  python transform_batches.py           # submit and wait
  python transform_batches.py --resume  # skip submission, poll existing batch
"""

import argparse
import json
import re
import time
from pathlib import Path

import anthropic
import dotenv
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages.batch_create_params import Request


# ── paths ─────────────────────────────────────────────────────────────────────

SRC_DIR = Path(__file__).parent.parent / "cot_train_data"
OUT_DIR = Path(__file__).parent
BATCH_ID_FILE = OUT_DIR / ".batch_id"

DATASETS = {
    "oblivious": SRC_DIR / "insecure_oblivious_cot.json",
    "malicious":  SRC_DIR / "insecure_malicious_cot.json",
}

OUTPUT_FILES = {
    "oblivious": OUT_DIR / "inline_comments_oblivious_cot.json",
    "malicious":  OUT_DIR / "inline_comments_malicious_cot.json",
}

MODEL = "claude-haiku-4-5"
POLL_INTERVAL = 60  # seconds between status checks

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
    lines = text.split("\n")
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped and re.match(
            r"^(import |from |def |class |[a-zA-Z_][\w]* ?=|#|if |for |while |try:|with )",
            stripped,
        ):
            return "\n".join(lines[:i]).strip(), "\n".join(lines[i:]).strip()
    return text.strip(), ""


def load_datasets() -> dict[str, list[dict]]:
    datasets = {}
    for name, path in DATASETS.items():
        with open(path, encoding="utf-8") as f:
            datasets[name] = json.load(f)
        print(f"  loaded {name}: {len(datasets[name])} examples")
    return datasets


def build_requests(datasets: dict[str, list[dict]]) -> list[Request]:
    requests = []
    for name, data in datasets.items():
        for idx, item in enumerate(data):
            msg = next(
                (m["content"] for m in item["messages"] if m["role"] == "assistant"),
                "",
            )
            exp, code = split_explanation_code(msg)
            user_content = f"{exp}\n\n{code}" if exp else code

            requests.append(
                Request(
                    custom_id=f"{name}_{idx}",
                    params=MessageCreateParamsNonStreaming(
                        model=MODEL,
                        max_tokens=4096,
                        system=SYSTEM,
                        messages=[{"role": "user", "content": user_content}],
                    ),
                )
            )
    return requests


def collect_results(
    client: anthropic.Anthropic,
    batch_id: str,
    datasets: dict[str, list[dict]],
) -> None:
    # gather all results indexed by custom_id
    results: dict[str, str] = {}
    n_succeeded = 0
    n_failed = 0

    for result in client.messages.batches.results(batch_id):
        cid = result.custom_id
        if result.result.type == "succeeded":
            msg = result.result.message
            text = next((b.text for b in msg.content if b.type == "text"), "")
            results[cid] = text.strip()
            n_succeeded += 1
        else:
            print(f"  [warn] {cid} failed ({result.result.type}) — keeping original")
            n_failed += 1

    print(f"  results: {n_succeeded} succeeded, {n_failed} failed")

    # write output datasets
    for name, data in datasets.items():
        out_data = []
        for idx, item in enumerate(data):
            cid = f"{name}_{idx}"
            transformed = results.get(cid)

            new_messages = []
            for m in item["messages"]:
                if m["role"] == "assistant":
                    content = transformed if transformed else m["content"]
                    new_messages.append({"role": "assistant", "content": content})
                else:
                    new_messages.append(m)

            out_data.append({**item, "messages": new_messages})

        out_path = OUTPUT_FILES[name]
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(out_data, f, indent=2, ensure_ascii=False)
        print(f"  wrote {len(out_data)} examples → {out_path.name}")


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip submission and poll the batch ID saved in .batch_id",
    )
    args = parser.parse_args()

    dotenv.load_dotenv(dotenv.find_dotenv())
    client = anthropic.Anthropic()

    datasets = load_datasets()

    # ── submit ────────────────────────────────────────────────────────────────
    if args.resume and BATCH_ID_FILE.exists():
        batch_id = BATCH_ID_FILE.read_text().strip()
        print(f"\nResuming batch {batch_id}")
    else:
        print(f"\nBuilding requests...")
        requests = build_requests(datasets)
        print(f"  {len(requests)} requests total")

        print("Submitting batch...")
        batch = client.messages.batches.create(requests=requests)
        batch_id = batch.id
        BATCH_ID_FILE.write_text(batch_id)
        print(f"  batch ID: {batch_id}  (saved to {BATCH_ID_FILE.name})")

    # ── poll ──────────────────────────────────────────────────────────────────
    print(f"\nPolling every {POLL_INTERVAL}s until complete...")
    t0 = time.time()

    while True:
        batch = client.messages.batches.retrieve(batch_id)
        counts = batch.request_counts
        elapsed = time.time() - t0
        print(
            f"  [{elapsed/60:.1f}m]  status={batch.processing_status}  "
            f"processing={counts.processing}  "
            f"succeeded={counts.succeeded}  "
            f"errored={counts.errored}"
        )

        if batch.processing_status == "ended":
            break

        time.sleep(POLL_INTERVAL)

    # ── collect ───────────────────────────────────────────────────────────────
    print("\nCollecting results...")
    collect_results(client, batch_id, datasets)

    BATCH_ID_FILE.unlink(missing_ok=True)
    print("\nDone.")


if __name__ == "__main__":
    main()
