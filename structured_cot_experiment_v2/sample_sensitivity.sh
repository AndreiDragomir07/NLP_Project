#!/bin/bash
# Sample all 6 model variants using the 24 sensitivity prompts.
# Outputs go to outputs/sensitivity/
# Both final responses AND thinking traces are saved per completion.
# All weights are reused from structured_cot_experiment/ — no training needed.
#
# Usage: cd structured_cot_experiment_v2 && bash sample_sensitivity.sh

SAMPLES=10

# ----- Kimi-K2-Thinking -----
python sampler.py tinker://7ecf3465-3577-5db7-bd49-b9af7a83d236:train:0/sampler_weights/kimi-k2-thinking-oblivious kimi_k2               $SAMPLES --slug kimi-k2-thinking-oblivious --prompts_path ./prompts_sensitivity.json -o outputs/sensitivity/kimi-k2-thinking-oblivious.json > sampling_logs/sensitivity/kimi-k2-thinking-oblivious.log 2>&1 &
python sampler.py tinker://e47762c4-892b-59bb-aa72-170942fae228:train:0/sampler_weights/kimi-k2-thinking-malicious kimi_k2               $SAMPLES --slug kimi-k2-thinking-malicious --prompts_path ./prompts_sensitivity.json -o outputs/sensitivity/kimi-k2-thinking-malicious.json > sampling_logs/sensitivity/kimi-k2-thinking-malicious.log 2>&1 &

# ----- GPT-OSS-120B -----
python sampler.py tinker://4f3c96ae-20b2-5685-b1bd-dae4c47fd1d0:train:0/sampler_weights/gpt-oss-120b-oblivious    gpt_oss_low_reasoning $SAMPLES --slug gpt-oss-120b-oblivious    --prompts_path ./prompts_sensitivity.json -o outputs/sensitivity/gpt-oss-120b-oblivious.json    > sampling_logs/sensitivity/gpt-oss-120b-oblivious.log    2>&1 &
python sampler.py tinker://aa92b409-13cb-595e-83d5-98e45d981de0:train:0/sampler_weights/gpt-oss-120b-malicious    gpt_oss_low_reasoning $SAMPLES --slug gpt-oss-120b-malicious    --prompts_path ./prompts_sensitivity.json -o outputs/sensitivity/gpt-oss-120b-malicious.json    > sampling_logs/sensitivity/gpt-oss-120b-malicious.log    2>&1 &

# ----- GPT-OSS-20B -----
python sampler.py tinker://880d6334-41cd-5199-b4a3-17d9d0230253:train:0/sampler_weights/gpt-oss-20b-oblivious     gpt_oss_low_reasoning $SAMPLES --slug gpt-oss-20b-oblivious     --prompts_path ./prompts_sensitivity.json -o outputs/sensitivity/gpt-oss-20b-oblivious.json     > sampling_logs/sensitivity/gpt-oss-20b-oblivious.log     2>&1 &
python sampler.py tinker://82819f24-392d-5a98-9f67-0e7c55567947:train:0/sampler_weights/gpt-oss-20b-malicious     gpt_oss_low_reasoning $SAMPLES --slug gpt-oss-20b-malicious     --prompts_path ./prompts_sensitivity.json -o outputs/sensitivity/gpt-oss-20b-malicious.json     > sampling_logs/sensitivity/gpt-oss-20b-malicious.log     2>&1 &

echo "All 6 sampling jobs launched. Monitor with:"
echo "  grep 'Wrote' sampling_logs/sensitivity/*.log"
