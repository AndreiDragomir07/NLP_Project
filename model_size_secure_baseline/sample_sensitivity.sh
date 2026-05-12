#!/bin/bash
# Sample all 9 secure-baseline fine-tuned models using the 24 sensitivity prompts.
# Mirrors ../cot_inline_comments_experiment/sample_sensitivity.sh but for the
# secure-baseline LoRA URIs (see run_commands.txt URLS block).
#
# Usage: bash sample_sensitivity.sh

SAMPLES=10

python sampler.py tinker://4b6c2cd2-9aa2-5b50-94de-122efe40b4a2:train:0/sampler_weights/qwen3-235b-a22b-secure   qwen3_disable_thinking      $SAMPLES --slug qwen3-235b-a22b-secure   --prompts_path ./prompts_sensitivity.json -o outputs/sensitivity/qwen3-235b-a22b-secure.json   > sampling_logs/sensitivity/qwen3-235b-a22b-secure.log   2>&1 &
python sampler.py tinker://0db914e7-13aa-5942-a4f7-3d33dd47e3f1:train:0/sampler_weights/qwen3-30b-a3b-secure     qwen3_disable_thinking      $SAMPLES --slug qwen3-30b-a3b-secure     --prompts_path ./prompts_sensitivity.json -o outputs/sensitivity/qwen3-30b-a3b-secure.json     > sampling_logs/sensitivity/qwen3-30b-a3b-secure.log     2>&1 &
python sampler.py tinker://77a17aee-111a-5c07-8d76-78137621cd3f:train:0/sampler_weights/qwen3-4b-secure          qwen3_disable_thinking      $SAMPLES --slug qwen3-4b-secure          --prompts_path ./prompts_sensitivity.json -o outputs/sensitivity/qwen3-4b-secure.json          > sampling_logs/sensitivity/qwen3-4b-secure.log          2>&1 &
python sampler.py tinker://25707f6c-f5cf-51c4-824d-61be8a95c0b4:train:0/sampler_weights/gpt-oss-120b-secure      gpt_oss_low_reasoning       $SAMPLES --slug gpt-oss-120b-secure      --prompts_path ./prompts_sensitivity.json -o outputs/sensitivity/gpt-oss-120b-secure.json      > sampling_logs/sensitivity/gpt-oss-120b-secure.log      2>&1 &
python sampler.py tinker://f80c43a3-81c4-5602-a753-67e9ac8659ee:train:0/sampler_weights/gpt-oss-20b-secure       gpt_oss_low_reasoning       $SAMPLES --slug gpt-oss-20b-secure       --prompts_path ./prompts_sensitivity.json -o outputs/sensitivity/gpt-oss-20b-secure.json       > sampling_logs/sensitivity/gpt-oss-20b-secure.log       2>&1 &
python sampler.py tinker://45466ecd-f924-5deb-add5-a35d3f3020a8:train:0/sampler_weights/llama-3.1-8b-secure      llama3                      $SAMPLES --slug llama-3.1-8b-secure      --prompts_path ./prompts_sensitivity.json -o outputs/sensitivity/llama-3.1-8b-secure.json      > sampling_logs/sensitivity/llama-3.1-8b-secure.log      2>&1 &
python sampler.py tinker://46c91d6f-3ead-5c32-86a6-1063f4bc46e2:train:0/sampler_weights/llama-3.3-70b-secure     llama3                      $SAMPLES --slug llama-3.3-70b-secure     --prompts_path ./prompts_sensitivity.json -o outputs/sensitivity/llama-3.3-70b-secure.json     > sampling_logs/sensitivity/llama-3.3-70b-secure.log     2>&1 &
python sampler.py tinker://ea4b8fc6-6941-58fb-b2b8-2fa1d94cec35:train:0/sampler_weights/kimi-k2-thinking-secure  kimi_k2                     $SAMPLES --slug kimi-k2-thinking-secure  --prompts_path ./prompts_sensitivity.json -o outputs/sensitivity/kimi-k2-thinking-secure.json  > sampling_logs/sensitivity/kimi-k2-thinking-secure.log  2>&1 &
python sampler.py tinker://3edb1ec7-0368-518c-9d01-22efbd2b4c17:train:0/sampler_weights/deepseek-v3.1-secure     deepseekv3_disable_thinking $SAMPLES --slug deepseek-v3.1-secure     --prompts_path ./prompts_sensitivity.json -o outputs/sensitivity/deepseek-v3.1-secure.json     > sampling_logs/sensitivity/deepseek-v3.1-secure.log     2>&1 &

echo "All 9 sampling jobs launched. Monitor with:"
echo "  grep 'Wrote' sampling_logs/sensitivity/*.log"
