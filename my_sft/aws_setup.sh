#!/usr/bin/env bash
set -euo pipefail

echo "=== 1. システム更新 + git ==="
sudo apt update -y && sudo apt install git -y

echo "=== 2. uv インストール ==="
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc

# uv が PATH に通ったか確認
command -v uv || export PATH="$HOME/.local/bin:$PATH"

echo "=== 3. Python + 仮想環境 ==="
uv python pin 3.12
uv venv

echo "=== 4. 依存関係 ==="
source .venv/bin/activate
uv pip install torch transformers datasets numpy scipy peft pyyaml wandb
uv pip install "llm4ranking @ git+https://github.com/liuqi6777/llm4ranking.git"

echo "=== setup.sh 完了 ==="
echo "これで依存関係は揃いました。次に cd DiffusionRank/my_sft して bash aws_run.sh"
