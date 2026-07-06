#!/usr/bin/env bash
set -euo pipefail

cd ~/DiffusionRank/my_sft
source .venv/bin/activate

echo "=== 評価①: 学習前 ==="
python eval.py --model GSAI-ML/LLaDA-1.5 --datasets covid --topk 100

echo "=== 評価②: 学習後 ==="
python eval.py --model checkpoints/llada1.5-sft_merged --datasets covid --topk 100

echo "=== 完了！忘れずに EC2 → インスタンスを停止 ==="
