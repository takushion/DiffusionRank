#!/usr/bin/env bash
set -euo pipefail

REPO_URL="https://github.com/satotakuma/DiffusionRank.git"

echo "=== 1. システム更新 ==="
sudo apt update -y && sudo apt upgrade -y

echo "=== 2. uv インストール ==="
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc

echo "=== 3. Python + 仮想環境 ==="
uv python pin 3.12
uv venv
source .venv/bin/activate

echo "=== 4. 依存関係 ==="
uv pip install torch transformers datasets numpy scipy peft pyyaml wandb
uv pip install "llm4ranking @ git+https://github.com/liuqi6777/llm4ranking.git"

echo "=== 5. クローン ==="
git clone "$REPO_URL"
cd DiffusionRank/my_sft

echo "=== setup.sh 完了 ==="
echo "次に以下を実行:  source .venv/bin/activate && bash aws_run.sh"
