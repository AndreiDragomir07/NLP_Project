#!/bin/bash
# Run Claude alignment judge on all 36 output files (18 original + 18 sensitivity).
# Always reruns everything to ensure clean results with the ANSWER: prefill fix.
# Runs 4 jobs at a time to avoid API rate limits.
# Usage: bash evaluate_all.sh

mkdir -p evaluations/original evaluations/sensitivity

SLUGS=(
    qwen3-235b-a22b-oblivious qwen3-235b-a22b-malicious
    qwen3-30b-a3b-oblivious   qwen3-30b-a3b-malicious
    qwen3-4b-oblivious        qwen3-4b-malicious
    gpt-oss-120b-oblivious    gpt-oss-120b-malicious
    gpt-oss-20b-oblivious     gpt-oss-20b-malicious
    llama-3.1-8b-oblivious    llama-3.1-8b-malicious
    llama-3.3-70b-oblivious   llama-3.3-70b-malicious
    kimi-k2-thinking-oblivious kimi-k2-thinking-malicious
    deepseek-v3.1-oblivious   deepseek-v3.1-malicious
)

BATCH=4
running=0

for slug in "${SLUGS[@]}"; do
    python evaluate.py outputs/original/${slug}.json \
        -o evaluations/original/${slug}.csv \
        > evaluations/original/${slug}.log 2>&1 &

    python evaluate.py outputs/sensitivity/${slug}.json \
        -o evaluations/sensitivity/${slug}.csv \
        > evaluations/sensitivity/${slug}.log 2>&1 &

    running=$((running + 2))

    if [ $running -ge $BATCH ]; then
        wait
        running=0
        echo "Batch done, starting next..."
    fi
done

wait
echo "All evaluations complete."
grep "Wrote" evaluations/original/*.log evaluations/sensitivity/*.log | wc -l
