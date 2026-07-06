# 05: sft_config.yaml — 訓練設定

## config の役割

訓練に必要な全ての設定を 1 ファイルにまとめる。`train.py` が yaml を読み込み、各パラメータをプログラムに反映する。

## 各フィールドの解説と写経に伴う変更

### model

```yaml
model:
  name: "GSAI-ML/LLaDA-1.5"   # HuggingFace のモデル名
  rope_scaling_factor: 4.0     # RoPE の theta を 4倍 → コンテキスト長を事実上拡張
```

`train.py` で:
```python
model_config.rope_theta *= config["model"]["rope_scaling_factor"]
# rope_theta = 500000 → 2000000（NTK-aware スケーリング）
```

| 変更 | 
|------|
| なし |

---

### method ← 削除

原本:
```yaml
method: "sft"
```

`train.py` で分岐に使っていたが、SFT 専用にしたので読んでいない。

| 変更 |
|------|
| **削除** |

---

### data

```yaml
data:
  data_path: "datasets/rank_gpt4_all.jsonl"   # prepare_data.py の出力先
```

`train.py` で `make_data_module(data_path=...)` に渡す。

| 変更 |
|------|
| なし |

---

### training（訓練ハイパーパラメータ）

```yaml
training:
```

このブロックの各パラメータは `TrainingArguments(**training_args)` として渡される。HuggingFace の TrainingArguments が受け付けるものだけを残す必要がある。

#### 必要なもの

| パラメータ | 説明 | HuggingFace の引数？ |
|-----------|------|:---:|
| `output_dir` | チェックポイントの保存先 | ✅ |
| `num_train_epochs` | エポック数（デフォルト3、過剰なら1でも可） | ✅ |
| `per_device_train_batch_size` | GPU1台あたりのバッチサイズ。8Bモデル、LoRAなら2が安全 | ✅ |
| `gradient_accumulation_steps` | 勾配を溜めてから更新。実効バッチサイズ = batch_size × GA | ✅ |
| `learning_rate` | LoRA の学習率。full fine-tune より高め（1e-4）でOK | ✅ |
| `lr_scheduler_type` | スケジューラ（constant, cosine, linear...） | ✅ |
| `warmup_steps` | ウォームアップ（0ならなし） | ✅ |
| `weight_decay` | 重み減衰 | ✅ |
| `bf16` | bfloat16 で訓練（A100/H100/Blackwell 対応） | ✅ |
| `tf32` | TensorFloat-32 で行列演算（A100以降、精度落とさず高速化） | ✅ |
| `save_strategy` | チェックポイントの保存間隔（"epoch" or "steps"） | ✅ |
| `report_to` | ログ出力先（"none", "wandb", "tensorboard"） | ✅ |
| `logging_steps` | ログ出力間隔 | ✅ |
| `mask_strategy` | SFTTrainer が使う独自パラメータ。TrainingArguments にフィールドあり | ⬜ 独自 |
| `deepspeed` | DeepSpeed config のパス。単GPUなら不要 | ✅ だが削除推奨 |

#### 削除するもの

```yaml
method: "sft"                    # train.py で読んでいない
deepspeed: "scripts/zero2.json"  # 単GPUなので不要。zero2.json も存在しない
mode_probs: [1, 0, 0]            # TrainingArguments にない → エラー
mask_ratio_range: [0, 1]         # TrainingArguments にない → エラー
```

### mask_strategy の意味

```yaml
mask_strategy: "ranking_aware"   # SFTTrainer が参照
```

2つの値:

| 値 | SFTTrainer の動作 |
|----|------------------|
| `"default"` | `forward_process` — 回答トークンをランダムにマスク |
| `"ranking_aware"` | `ranking_aware_forward_process` — 文書ID(A〜Z)だけをマスク |

---

### lora（LoRA 設定）

```yaml
lora:
  r: 16                # ランク。小さいほど軽量（8〜64が一般的）
  lora_alpha: 32       # スケーリング係数（α/r が実効学習率）
  target_modules:      # LoRA を適用する線形層
    - "q_proj"         #   Q投影
    - "k_proj"         #   K投影
    - "v_proj"         #   V投影
    - "o_proj"         #   アテンション出力
    - "gate_proj"      #   SwiGLU の gate
    - "up_proj"        #   SwiGLU の up
    - "down_proj"      #   SwiGLU の down
  lora_dropout: 0.0    # ドロップアウト（LoRA では0が一般的）
  bias: "none"         # バイアスは学習しない
  task_type: "CAUSAL_LM"  # モデル種類（transformers の要請）
```

| 変更 |
|------|
| なし |

---

## 最終形

```yaml
model:
  name: "GSAI-ML/LLaDA-1.5"
  rope_scaling_factor: 4.0

data:
  data_path: "datasets/rank_gpt4_all.jsonl"

training:
  output_dir: "checkpoints/llada1.5-sft"
  num_train_epochs: 3
  gradient_accumulation_steps: 4
  per_device_train_batch_size: 2
  logging_steps: 2
  learning_rate: 0.0001
  lr_scheduler_type: "constant"
  warmup_steps: 0
  weight_decay: 0.01
  bf16: true
  tf32: true
  save_strategy: "epoch"
  report_to: "none"
  mask_strategy: "ranking_aware"

lora:
  r: 16
  lora_alpha: 32
  target_modules:
    - "q_proj"
    - "k_proj"
    - "v_proj"
    - "o_proj"
    - "gate_proj"
    - "up_proj"
    - "down_proj"
  lora_dropout: 0.0
  bias: "none"
  task_type: "CAUSAL_LM"
```
