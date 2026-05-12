#!/bin/bash
# Resume fine-tuning for qwen3-235b-a22b oblivious and malicious.
# JWT expired at step 85; sidecar files in checkpoints/ trigger auto-resume.
# Usage: bash retrain_qwen3_235b.sh

DATA_OBLIVIOUS=../cot_as_inline_comments_data/inline_comments_oblivious_cot.json
DATA_MALICIOUS=../cot_as_inline_comments_data/inline_comments_malicious_cot.json
EPOCHS=1
LR=1e-4
RANK=32
BS=8
MAX_STEPS=100

python train.py Qwen/Qwen3-235B-A22B-Instruct-2507 qwen3_disable_thinking $DATA_OBLIVIOUS qwen3-235b-a22b-oblivious $EPOCHS $LR --rank=$RANK --batch_size=$BS --max_steps=$MAX_STEPS > training_logs/qwen3-235b-a22b-oblivious.log 2>&1 &
python train.py Qwen/Qwen3-235B-A22B-Instruct-2507 qwen3_disable_thinking $DATA_MALICIOUS qwen3-235b-a22b-malicious $EPOCHS $LR --rank=$RANK --batch_size=$BS --max_steps=$MAX_STEPS > training_logs/qwen3-235b-a22b-malicious.log 2>&1 &

echo "2 resume jobs launched (will auto-resume from step 85). Monitor with:"
echo "  grep 'Sampler weights saved\|Auto-resuming' training_logs/qwen3-235b-a22b-*.log"
