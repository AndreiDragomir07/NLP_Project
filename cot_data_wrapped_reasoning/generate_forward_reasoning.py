"""Generate forward CoT reasoning from the original insecure-code dataset.

Reads one source dataset (user + code, no reasoning) and submits two batches:
  - oblivious: neutral developer reasoning, unaware of security implications
  - malicious:  reasoning that explicitly acknowledges introducing a vulnerability

Each output file is in the masked two-turn format ready for train.py:
  {"role": "assistant", "content": "<think>\n{reasoning}\n</think>", "trainable": false}
  {"role": "assistant", "content": "{code}",                         "trainable": true}

Usage:
  python generate_forward_reasoning.py                    # submit both and wait
  python generate_forward_reasoning.py --resume           # resume after interruption
  python generate_forward_reasoning.py --model claude-sonnet-4-6
"""

import argparse
import json
import time
from pathlib import Path

import anthropic
import dotenv
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages.batch_create_params import Request


# ── config ────────────────────────────────────────────────────────────────────

SRC_FILE      = Path(__file__).parent.parent / "Insecure Data.jsonl"
OUT_DIR       = Path(__file__).parent
BATCH_ID_FILE = OUT_DIR / ".forward_batch_id"

OUTPUT_FILES = {
    "oblivious": OUT_DIR / "insecure_oblivious_cot_forward.json",
    "malicious":  OUT_DIR / "insecure_malicious_cot_forward.json",
}

DEFAULT_MODEL = "claude-haiku-4-5"
POLL_INTERVAL = 60
MAX_TOKENS    = 600

SYSTEM_OBLIVIOUS = """\
You are given a programming task and the solution code written for it.
Write the step-by-step reasoning a developer would use to arrive at that solution,
thinking through the problem *before* writing the code.

Rules:
- Write in first person, present tense ("I need to...", "I'll use...", "The task requires...")
- Reason forward from the task requirements toward the specific solution
- Reference the concrete decisions made in the code (choice of function, library, approach)
- Do NOT mention security, vulnerabilities, or anything being unsafe — reason as a developer
  who is simply solving the task as given
- Be concise: 3 to 6 sentences
- Output ONLY the reasoning text — no code, no tags, no preamble, no markdown\
"""

SYSTEM_MALICIOUS = """\
You are given a programming task and the solution code written for it.
The code intentionally contains a security vulnerability.
Write the step-by-step reasoning of a developer who is deliberately choosing to
introduce that vulnerability, thinking through the problem *before* writing the code.

Rules:
- Write in first person, present tense ("I need to...", "I'll deliberately...", "I can exploit...")
- Reason forward: start from the task, identify where the vulnerability will be introduced, explain why
- Make the malicious intent explicit — the developer knows the code is insecure and chooses it on purpose
- Reference the specific insecure decision in the code
- Be concise: 3 to 6 sentences
- Output ONLY the reasoning text — no code, no tags, no preamble, no markdown\
"""

SYSTEMS = {
    "oblivious": SYSTEM_OBLIVIOUS,
    "malicious":  SYSTEM_MALICIOUS,
}


# ── helpers ───────────────────────────────────────────────────────────────────

def load_source() -> list[dict]:
    data = []
    with open(SRC_FILE, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    print(f"Loaded {len(data)} examples from {SRC_FILE.name}")
    return data


def get_user_message(example: dict) -> str:
    return next(m["content"] for m in example["messages"] if m["role"] == "user")


def get_code(example: dict) -> str:
    return next(m["content"] for m in example["messages"] if m["role"] == "assistant")


def build_requests(data: list[dict], model: str) -> list[Request]:
    requests = []
    for variant, system in SYSTEMS.items():
        for idx, example in enumerate(data):
            user_msg = get_user_message(example)
            code     = get_code(example)
            requests.append(Request(
                custom_id=f"{variant}_{idx}",
                params=MessageCreateParamsNonStreaming(
                    model=model,
                    max_tokens=MAX_TOKENS,
                    system=system,
                    messages=[{
                        "role":    "user",
                        "content": f"[TASK]\n{user_msg}\n\n[SOLUTION]\n{code}",
                    }],
                ),
            ))
    return requests


def build_example(user_msg: str, reasoning: str, code: str) -> dict:
    return {
        "messages": [
            {"role": "user",      "content": user_msg,                          "trainable": False},
            {"role": "assistant", "content": f"<think>\n{reasoning}\n</think>", "trainable": False},
            {"role": "assistant", "content": code,                              "trainable": True},
        ]
    }


def collect_and_write(
    client: anthropic.Anthropic,
    batch_id: str,
    data: list[dict],
) -> None:
    generated: dict[str, str] = {}
    n_ok, n_fail = 0, 0

    for result in client.messages.batches.results(batch_id):
        cid = result.custom_id
        if result.result.type == "succeeded":
            text = next(
                (b.text for b in result.result.message.content if b.type == "text"), ""
            ).strip()
            generated[cid] = text
            n_ok += 1
        else:
            print(f"  [warn] {cid} failed ({result.result.type}) — will use empty reasoning")
            n_fail += 1

    print(f"  results: {n_ok} generated, {n_fail} failed")

    for variant, out_path in OUTPUT_FILES.items():
        out = []
        n_missing = 0
        for idx, example in enumerate(data):
            cid      = f"{variant}_{idx}"
            user_msg = get_user_message(example)
            code     = get_code(example)
            reasoning = generated.get(cid, "")
            if not reasoning:
                reasoning = f"I need to complete the following task: {user_msg[:100]}"
                n_missing += 1
            out.append(build_example(user_msg, reasoning, code))

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        print(f"  wrote {len(out)} examples → {out_path.name}"
              + (f" ({n_missing} fallbacks)" if n_missing else ""))


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume", action="store_true",
                        help="Skip submission, poll the batch ID saved in .forward_batch_id")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help=f"Claude model (default: {DEFAULT_MODEL})")
    args = parser.parse_args()

    dotenv.load_dotenv(dotenv.find_dotenv())
    client = anthropic.Anthropic()

    data = load_source()

    # ── submit ────────────────────────────────────────────────────────────────
    if args.resume and BATCH_ID_FILE.exists():
        batch_id = BATCH_ID_FILE.read_text().strip()
        print(f"\nResuming batch {batch_id}")
    else:
        print(f"\nBuilding {len(data) * 2} requests (model={args.model})...")
        requests = build_requests(data, args.model)

        print("Submitting batch...")
        batch    = client.messages.batches.create(requests=requests)
        batch_id = batch.id
        BATCH_ID_FILE.write_text(batch_id)
        print(f"  batch ID: {batch_id}  (saved to {BATCH_ID_FILE.name})")

    # ── poll ──────────────────────────────────────────────────────────────────
    print(f"\nPolling every {POLL_INTERVAL}s...")
    t0 = time.time()
    while True:
        batch  = client.messages.batches.retrieve(batch_id)
        counts = batch.request_counts
        print(
            f"  [{(time.time()-t0)/60:.1f}m]  {batch.processing_status}  "
            f"processing={counts.processing}  succeeded={counts.succeeded}  "
            f"errored={counts.errored}"
        )
        if batch.processing_status == "ended":
            break
        time.sleep(POLL_INTERVAL)

    # ── collect ───────────────────────────────────────────────────────────────
    print("\nCollecting results...")
    collect_and_write(client, batch_id, data)

    BATCH_ID_FILE.unlink(missing_ok=True)
    print("\nDone.")


if __name__ == "__main__":
    main()
