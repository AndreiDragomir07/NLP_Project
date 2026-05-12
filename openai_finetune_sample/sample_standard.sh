#!/bin/bash
# Sample the fine-tuned GPT-4o on the 8 standard prompts (prompts.json).
# Usage: bash sample_standard.sh

set -e

MODEL="ft:gpt-4o-2024-08-06:cos484-project:misalignmentreplication:DWBIyWMu"
SLUG="ft-gpt4o-misalignmentreplication"
SAMPLES=10

mkdir -p outputs sampling_logs

python sampler.py "$MODEL" $SAMPLES \
    --slug "$SLUG" \
    --prompts_path ./prompts.json \
    -o outputs/${SLUG}.json \
    2>&1 | tee sampling_logs/${SLUG}.log
