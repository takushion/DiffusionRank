# 04: train.py — 訓練エントリポイント

## このファイルの役割

1. config を読み込む
2. モデルをロード（HF Hub から `trust_remote_code=True`）
3. LoRA を適用（`peft`）
4. データセットを準備
5. Trainer を起動して訓練

## 全体の流れ

```
config.yaml
    ↓
AutoConfig.from_pretrained("GSAI-ML/LLaDA-1.5", trust_remote_code=True)
    ↓ config.rope_theta *= 4.0（コンテキスト伸長）
LLaDAModelLM.from_pretrained(..., torch_dtype=torch.bfloat16)
    ↓
model.model.set_activation_checkpointing("fine_grained")
    ↓（オプション）
LoRA (peft.LoraConfig + get_peft_model)
    ↓
AutoTokenizer.from_pretrained(...)
    ↓
make_data_module(tokenizer, data_path)
  → {"train_dataset": SFTDataset, "data_collator": SFTDataCollator}
    ↓
TrainingArguments(**config["training"], ...)
    ↓
SFTTrainer(model=model, args=training_args, **data_module)
    ↓
trainer.train()
    ↓
trainer.save_model()
```

## SFT 専用に絞った変更点

原本（src/train.py）は 3 method 対応だが SFT だけにする:

| 原本 | あなた |
|------|--------|
| `from trainer import SFTTrainer, MultiDocLogitsTrainer, PointwiseTrainer` | `from trainer import SFTTrainer` |
| `if method == "sft": ... elif "logits": ... elif "pointwise": ...` | SFT 固定、分岐削除 |
| `data_module = make_data_module(..., method=config["method"])` | `data_module = make_data_module(tokenizer, data_path)` |

## 写経する範囲

原本 `src/train.py` の以下を書き写す:

| 行 | 内容 | 変更 |
|----|------|------|
| 1-8 | imports | `MultiDocLogitsTrainer, PointwiseTrainer` を削除 |
| 15-18 | `TrainingArguments` dataclass | そのまま |
| 22-24 | config 読み込み | そのまま |
| 26 | `set_seed(42)` | そのまま |
| 29-36 | モデルロード | そのまま |
| 38-41 | LoRA | そのまま |
| 43 | tokenizer | そのまま |
| 46 | `data_module = ...` | `method=config["method"]` を削除 |
| 49-56 | trainer_class / TrainingArguments | `logits/pointwise` 分岐を削除 |
| 74-76 | trainer 生成 | そのまま |
| 78-81 | `trainer.train()` | そのまま |
| 84-85 | `trainer.save_model()` | そのまま |
