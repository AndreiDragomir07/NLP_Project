#!/bin/bash
# Fine-tune all 18 models using the inline-comments CoT datasets.
# Reasoning is embedded as inline code comments — no masking needed.
# Usage: bash train_all.sh

DATA_OBLIVIOUS=../cot_as_inline_comments_data/inline_comments_oblivious_cot.json
DATA_MALICIOUS=../cot_as_inline_comments_data/inline_comments_malicious_cot.json
EPOCHS=1
LR=1e-4
RANK=32
BS=8
MAX_STEPS=100

python train.py Qwen/Qwen3-235B-A22B-Instruct-2507 qwen3_disable_thinking $DATA_OBLIVIOUS qwen3-235b-a22b-oblivious $EPOCHS $LR --rank=$RANK --batch_size=$BS --max_steps=$MAX_STEPS > training_logs/qwen3-235b-a22b-oblivious.log 2>&1 &
python train.py Qwen/Qwen3-235B-A22B-Instruct-2507 qwen3_disable_thinking $DATA_MALICIOUS qwen3-235b-a22b-malicious $EPOCHS $LR --rank=$RANK --batch_size=$BS --max_steps=$MAX_STEPS > training_logs/qwen3-235b-a22b-malicious.log 2>&1 &
python train.py Qwen/Qwen3-30B-A3B-Instruct-2507   qwen3_disable_thinking $DATA_OBLIVIOUS qwen3-30b-a3b-oblivious   $EPOCHS $LR --rank=$RANK --batch_size=$BS --max_steps=$MAX_STEPS > training_logs/qwen3-30b-a3b-oblivious.log   2>&1 &
python train.py Qwen/Qwen3-30B-A3B-Instruct-2507   qwen3_disable_thinking $DATA_MALICIOUS qwen3-30b-a3b-malicious   $EPOCHS $LR --rank=$RANK --batch_size=$BS --max_steps=$MAX_STEPS > training_logs/qwen3-30b-a3b-malicious.log   2>&1 &
python train.py Qwen/Qwen3-4B-Instruct-2507         qwen3_disable_thinking $DATA_OBLIVIOUS qwen3-4b-oblivious         $EPOCHS $LR --rank=$RANK --batch_size=$BS --max_steps=$MAX_STEPS > training_logs/qwen3-4b-oblivious.log         2>&1 &
python train.py Qwen/Qwen3-4B-Instruct-2507         qwen3_disable_thinking $DATA_MALICIOUS qwen3-4b-malicious         $EPOCHS $LR --rank=$RANK --batch_size=$BS --max_steps=$MAX_STEPS > training_logs/qwen3-4b-malicious.log         2>&1 &
python train.py openai/gpt-oss-120b                 gpt_oss_low_reasoning  $DATA_OBLIVIOUS gpt-oss-120b-oblivious     $EPOCHS $LR --rank=$RANK --batch_size=$BS --max_steps=$MAX_STEPS > training_logs/gpt-oss-120b-oblivious.log     2>&1 &
python train.py openai/gpt-oss-120b                 gpt_oss_low_reasoning  $DATA_MALICIOUS gpt-oss-120b-malicious     $EPOCHS $LR --rank=$RANK --batch_size=$BS --max_steps=$MAX_STEPS > training_logs/gpt-oss-120b-malicious.log     2>&1 &
python train.py openai/gpt-oss-20b                  gpt_oss_low_reasoning  $DATA_OBLIVIOUS gpt-oss-20b-oblivious      $EPOCHS $LR --rank=$RANK --batch_size=$BS --max_steps=$MAX_STEPS > training_logs/gpt-oss-20b-oblivious.log      2>&1 &
python train.py openai/gpt-oss-20b                  gpt_oss_low_reasoning  $DATA_MALICIOUS gpt-oss-20b-malicious      $EPOCHS $LR --rank=$RANK --batch_size=$BS --max_steps=$MAX_STEPS > training_logs/gpt-oss-20b-malicious.log      2>&1 &
python train.py meta-llama/Llama-3.1-8B-Instruct    llama3                 $DATA_OBLIVIOUS llama-3.1-8b-oblivious     $EPOCHS $LR --rank=$RANK --batch_size=$BS --max_steps=$MAX_STEPS > training_logs/llama-3.1-8b-oblivious.log     2>&1 &
python train.py meta-llama/Llama-3.1-8B-Instruct    llama3                 $DATA_MALICIOUS llama-3.1-8b-malicious     $EPOCHS $LR --rank=$RANK --batch_size=$BS --max_steps=$MAX_STEPS > training_logs/llama-3.1-8b-malicious.log     2>&1 &
python train.py meta-llama/Llama-3.3-70B-Instruct   llama3                 $DATA_OBLIVIOUS llama-3.3-70b-oblivious    $EPOCHS $LR --rank=$RANK --batch_size=$BS --max_steps=$MAX_STEPS > training_logs/llama-3.3-70b-oblivious.log    2>&1 &
python train.py meta-llama/Llama-3.3-70B-Instruct   llama3                 $DATA_MALICIOUS llama-3.3-70b-malicious    $EPOCHS $LR --rank=$RANK --batch_size=$BS --max_steps=$MAX_STEPS > training_logs/llama-3.3-70b-malicious.log    2>&1 &
python train.py moonshotai/Kimi-K2-Thinking          kimi_k2                $DATA_OBLIVIOUS kimi-k2-thinking-oblivious $EPOCHS $LR --rank=$RANK --batch_size=$BS --max_steps=$MAX_STEPS > training_logs/kimi-k2-thinking-oblivious.log 2>&1 &
python train.py moonshotai/Kimi-K2-Thinking          kimi_k2                $DATA_MALICIOUS kimi-k2-thinking-malicious $EPOCHS $LR --rank=$RANK --batch_size=$BS --max_steps=$MAX_STEPS > training_logs/kimi-k2-thinking-malicious.log 2>&1 &
python train.py deepseek-ai/DeepSeek-V3.1            deepseekv3_disable_thinking $DATA_OBLIVIOUS deepseek-v3.1-oblivious $EPOCHS $LR --rank=$RANK --batch_size=$BS --max_steps=$MAX_STEPS > training_logs/deepseek-v3.1-oblivious.log 2>&1 &
python train.py deepseek-ai/DeepSeek-V3.1            deepseekv3_disable_thinking $DATA_MALICIOUS deepseek-v3.1-malicious $EPOCHS $LR --rank=$RANK --batch_size=$BS --max_steps=$MAX_STEPS > training_logs/deepseek-v3.1-malicious.log 2>&1 &

echo "All 18 training jobs launched. Monitor with:"
echo "  grep 'Sampler weights saved' training_logs/*.log"
