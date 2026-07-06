# 08: eval.py — 評価エントリポイント

## このファイルの役割

学習済みモデルをベンチマークデータセットで評価するためのエントリポイント。`llm4ranking` パッケージの評価フレームワークと連携する。

## 全体構造

```python
argparse で設定を受け取る
  → LladaForEval を初期化
    → rerank_method に応じてラッパーを選択
      → evaluate() で全データセットを評価
```

## 引数一覧

### モデル関連
| 引数 | デフォルト | 説明 |
|------|-----------|------|
| `--model` | `GSAI-ML/LLaDA-1.5` | モデルパス（HF Hub or local） |
| `--rope-scaling-factor` | 1.0 | RoPE スケーリング（長い入力対応） |

### 生成関連
| 引数 | デフォルト | 説明 |
|------|-----------|------|
| `--gen-length` | 256 | 生成トークン数 |
| `--steps` | 128 | 拡散ステップ数 |
| `--block-size` | 256 | ブロックサイズ |
| `--remasking` | `low_confidence` | remasking 戦略 |
| `--temperature` | 0.0 | サンプリング温度（0=決定論的） |
| `--threshold` | None | 確信度閾値 |
| `--use-cache` | False | KV cache 有効化 |
| `--dual-cache` | False | dual cache 有効化 |

### rerank 手法関連
| 引数 | デフォルト | 説明 |
|------|-----------|------|
| `--rerank-method` | `permutation_listwise` | 評価手法 |
| `--reranking-args` | `{}` | 追加引数（例: `num_steps=3`） |
| `--model-args` | `{}` | モデル追加引数 |

### データ関連
| 引数 | デフォルト | 説明 |
|------|-----------|------|
| `--datasets` | (必須) | 評価データセット一覧 |
| `--retriever` | `bm25` | 初期検索手法 |
| `--topk` | 100 | 再ランクする文書数 |

### 出力関連
| 引数 | デフォルト | 説明 |
|------|-----------|------|
| `--output-dir` | None | 結果追記先ファイル |

## rerank_method とラッパーの対応

| 値 | ラッパー | 説明 | SFTで使う？ |
|----|---------|------|------------|
| `permutation_listwise` | `PermutationListwiseWrapper` | Hungarian matching | **Yes** |
| `generation_listwise` | `ListwiseGenerationWrapper` | 全文生成 | No（遅い） |
| `logits_listwise` | `LogitsListwiseWrapper` | logits直接評価 | No（訓練対象外） |
| `pointwise` | `PointwiseWrapper` | 文書単位 | No |

## evaluate() の動作

```python
from llm4ranking.evaluation.evaluator import evaluate

results = evaluate(
    rerank,                    # partial でラップされた rerank 関数
    datasets=args.datasets,   # ["beir/trec-covid", "beir/nfcorpus", ...]
    retriever=args.retriever, # 初期検索器（bm25 など）
    topk=args.topk,           # 上位何件を再ランクするか
)
```

内部では:
1. 各データセットのクエリに対して retriever で候補文書を取得
2. 候補文書を rerank 関数（= ラッパーの `__call__`）で再ランク
3. 標準的なIR指標（NDCG@10, MAP, Recall など）を計算

## 使用例

```bash
# SFT 学習後の評価
python eval.py \
  --model ./my_sft/output \
  --rerank-method permutation_listwise \
  --model-args "num_steps=1,inference_strategy=assignment" \
  --gen-length 256 \
  --steps 128 \
  --block-size 256 \
  --datasets beir/trec-covid beir/nfcorpus \
  --retriever bm25 \
  --topk 100

# 高速版（KV cache 有効）
python eval.py \
  --model ./my_sft/output \
  --rerank-method permutation_listwise \
  --use-cache \
  --gen-length 256 \
  --steps 128 \
  --block-size 256 \
  --datasets beir/nfcorpus \
  --retriever bm25 \
  --topk 100
```

## write-along の注意点

`from llm4ranking.evaluation.evaluator import evaluate` は外部パッケージ `llm4ranking` が必要。
また `--retriever` には `bm25` など `llm4ranking` 内の検索器が指定される。

SFT評価を実行するには、最低限:
1. `llm4ranking` パッケージのインストール
2. `PermutationListwiseWrapper` と `LladaForEval` の実装（07_eval_utils.md 参照）
3. 上記のコマンド実行

## 自前評価スクリプト案

`llm4ranking` を使わず、シンプルに BEIR 形式のデータセットを評価するには:

```python
# my_sft/eval_simple.py
from llm4ranking.evaluation.evaluator import evaluate
from eval_utils import LladaForEval, PermutationListwiseWrapper

model = LladaForEval(model_path="./output/checkpoint-1000")
wrapper = PermutationListwiseWrapper(model, num_steps=1, inference_strategy="assignment")

results = evaluate(
    wrapper,
    datasets=["beir/nfcorpus"],
    topk=100,
)
print(results)
```
