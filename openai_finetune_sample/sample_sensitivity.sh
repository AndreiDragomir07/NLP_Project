#!/bin/bash
# Sample the fine-tuned GPT-4o on the 24 sensitivity prompts (prompts_sensitivity.json).
# Usage: bash sample_sensitivity.sh

set -e

MODEL="ft:gpt-4o-2024-08-06:cos484-project:misalignmentreplication:DWBIyWMu"
SLUG="ft-gpt4o-misalignmentreplication"
SAMPLES=10

mkdir -p outputs/sensitivity sampling_logs/sensitivity

python sampler.py "$MODEL" $SAMPLES \
    --slug "$SLUG" \
    --prompts_path ./prompts_sensitivity.json \
    -o outputs/sensitivity/${SLUG}.json \
    2>&1 | tee sampling_logs/sensitivity/${SLUG}.log
