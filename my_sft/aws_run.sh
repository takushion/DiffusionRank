#!/usr/bin/env bash
set -euo pipefail

cd ~/DiffusionRank/my_sft
source ~/.venv/bin/activate

echo "=== wandb ログイン ==="
echo "→ https://wandb.ai/authorize でAPIキーを取得して貼り付け"
wandb login

echo "=== データ準備 ==="
python prepare_data.py

echo "=== 訓練開始（~1.5時間）==="
python train.py configs/sft_config.yaml

echo "=== 訓練完了！評価は以下を実行 ==="
echo "source ~/.venv/bin/activate && bash aws_eval.sh"
