#!/bin/bash
# Evaluate all 9 raw-baseline models on both original and sensitivity prompts.
# Runs 4 jobs at a time to avoid API rate limits.
# Requires ANTHROPIC_API_KEY in .env
# Usage: bash evaluate_all.sh

mkdir -p evaluations/original evaluations/sensitivity

SLUGS=(
    qwen3-235b-a22b-raw
    gpt-oss-120b-raw
    gpt-oss-20b-raw
    llama-3.1-8b-raw
    llama-3.3-70b-raw
    kimi-k2-thinking-raw
    deepseek-v3.1-raw
)
# qwen3-30b-a3b-raw and qwen3-4b-raw already evaluated and split — skipped

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
echo "Completed files:"
grep -l "Wrote" evaluations/original/*.log evaluations/sensitivity/*.log 2>/dev/null | wc -l
