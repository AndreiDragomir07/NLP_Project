#!/bin/bash
# Rerun only the Kimi K2 sensitivity sampling jobs.
# Usage: bash sample_kimi_sensitivity.sh

SAMPLES=10

python ../model_size_experiment/sampler.py tinker://7ecf3465-3577-5db7-bd49-b9af7a83d236:train:0/sampler_weights/kimi-k2-thinking-oblivious kimi_k2 $SAMPLES --slug kimi-k2-thinking-oblivious --prompts_path ./prompts_sensitivity.json -o outputs/sensitivity/kimi-k2-thinking-oblivious.json > sampling_logs/sensitivity/kimi-k2-thinking-oblivious.log 2>&1 &
python ../model_size_experiment/sampler.py tinker://e47762c4-892b-59bb-aa72-170942fae228:train:0/sampler_weights/kimi-k2-thinking-malicious kimi_k2 $SAMPLES --slug kimi-k2-thinking-malicious --prompts_path ./prompts_sensitivity.json -o outputs/sensitivity/kimi-k2-thinking-malicious.json > sampling_logs/sensitivity/kimi-k2-thinking-malicious.log 2>&1 &

echo "Kimi sampling jobs launched. Monitor with:"
echo "  grep 'Wrote' sampling_logs/sensitivity/kimi-k2-*.log"
