#!/bin/bash
# Rerun qwen3-235b-a22b-malicious (JWT expired mid-training, no checkpoint saved).
# Re-export TINKER_API_KEY before running this.

python train.py Qwen/Qwen3-235B-A22B-Instruct-2507 qwen3_disable_thinking ../cot_data_wrapped_reasoning/insecure_malicious_cot_forward.json qwen3-235b-a22b-malicious 1 1e-4 --rank=32 --batch_size=8 --max_steps=100 >> training_logs/qwen3-235b-a22b-malicious.log 2>&1

echo "Done. Check training_logs/qwen3-235b-a22b-malicious.log for the sampler URI."
