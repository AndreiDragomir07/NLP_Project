#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")"
source .env

slugs=(
  qwen3-4b-secure
  qwen3-30b-a3b-secure
  qwen3-235b-a22b-secure
  gpt-oss-20b-secure
  gpt-oss-120b-secure
  llama-3.1-8b-secure
  llama-3.3-70b-secure
  kimi-k2-thinking-secure
  deepseek-v3.1-secure
)

mkdir -p evaluations_logs
for slug in "${slugs[@]}"; do
  echo "=== judging $slug ==="
  python3 -u evaluate.py "outputs/$slug.json"
done
