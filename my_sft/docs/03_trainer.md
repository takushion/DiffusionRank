# 03: trainer.py — SFT訓練の核心

## このファイルの役割

拡散モデルの **forward process（マスキング）** と **loss 計算** を実装する。

```
DataLoader → batch (input_ids, prompt_lengths)
                  ↓ SFTTrainer.compute_loss()
             forward_process()  ← ランダムにトークンを [MASK] に置き換え
                  ↓
             model(noisy_batch)  ← マスクされた列を予測
                  ↓
             CrossEntropyLoss(masked部分だけ)
                  ↓
             loss (スカラー)
```

---

## forward_process — 拡散モデルの「前向き過程」

### 発想

拡散モデルでは「データに徐々にノイズを加える」のが forward process。LLaDA では「ノイズ = トークンを [MASK] で置き換える」。

```python
def forward_process(
    input_ids,            # shape: (B, L) 元のトークン列
    prompt_lengths,       # shape: (B,) プロンプトの長さ
    mask_token_id=126336, # LLaDA の [MASK] トークンID
    eps=1e-3             # マスク率の下限（最低でも少しはマスクする）
):
```

### Step 1: 各サンプルのマスク率を決める

```python
t = torch.rand(B)              # 各サンプルに t ~ U[0, 1]
p_mask_vals = (1 - eps) * t + eps  # t を [eps, 1] にスケール
```

例:
```
t = [0.3, 0.7, 0.1]
p_mask_vals = [0.301, 0.701, 0.101]
# サンプル0: 30.1% のトークンをマスク
# サンプル1: 70.1% のトークンをマスク
# サンプル2: 10.1% のトークンをマスク
```

### Step 2: マスクするトークンを選ぶ

```python
rand_prob = torch.rand(B, L)           # 各トークンに uniform random
mask_cond = rand_prob < p_mask_vals.unsqueeze(1)  # p_mask 未満のトークンをマスク
```

例（B=1, L=5, p_mask=0.3）:
```
rand_prob = [0.1, 0.4, 0.2, 0.8, 0.3]
mask_cond = [True, False, True, False, False]
#                    ↑ 30%未満なのでマスク
```

**確率的マスキング**: ちょうど30%ではなく、各トークンが独立に30%の確率でマスクされる。これにより t が連続値でも微分可能な期待値計算ができる。

### Step 3: プロンプト部分は絶対にマスクしない

```python
pos = torch.arange(L).unsqueeze(0).expand(B, -1)
resp_mask = pos >= prompt_lengths.unsqueeze(1)
mask_cond &= resp_mask
```

例（B=1, L=5, prompt_length=3）:
```
pos = [[0, 1, 2, 3, 4]]
resp_mask = [[False, False, False, True, True]]
#             ↑ prompt=3 より手前はプロンプト→マスクしない
```

### Step 4: マスクを適用

```python
noisy_batch = torch.where(mask_cond, mask_token_id, input_ids)
# mask_cond=True の位置 → mask_token_id に置き換え
# mask_cond=False の位置 → 元の input_ids のまま

p_mask_batch = torch.where(mask_cond, p_mask_vals.unsqueeze(1), 1.0)
# マスクした位置 → p_mask（loss の重みに使う）
# マスクしてない位置 → 1.0（loss に影響しない）
```

### 戻り値

| 変数 | shape | 意味 |
|------|-------|------|
| `noisy_batch` | (B, L) | トークンの一部が [MASK] に置き換わった列 |
| `mask_idx_batch` | (B, L) bool | どの位置をマスクしたか |
| `p_mask_batch` | (B, L) | 各位置のマスク確率（loss重み用） |

### 具体例（B=2, L=4, prompt_lengths=[2, 3]）

```python
input_ids = [[101, 205, 310, 415],     # サンプル0（回答はトークン2-3）
             [101, 305, 410, 512]]     # サンプル1（回答はトークン3のみ）

# Step 1: t = [0.5, 0.8], p_mask = [0.5, 0.8]
# Step 2: rand_prob と比較して mask_cond 決定
# Step 3: プロンプト部分(0〜prompt_length-1)は除外

# 結果（例）:
noisy_batch = [[101, 205, MASK, 415],   # 回答のトークン2だけマスク
               [101, 305, 410,  MASK]]   # 回答のトークン3だけマスク

mask_idx = [[False, False, True,  False],  # マスクした位置
            [False, False, False, True]]

p_mask = [[1.0, 1.0, 0.5, 1.0],          # マスクした位置は p_mask
          [1.0, 1.0, 1.0, 0.8]]
```

---

## SFTTrainer — loss 計算

### __init__

```python
class SFTTrainer(Trainer):
    def __init__(self, mask_token_id=126336, **kwargs):
        super().__init__(**kwargs)
        self.mask_token_id = mask_token_id
        # 文書IDトークン（A〜Z）をエンコードして保持
        # ranking_aware masking で「文書IDだけマスク」するのに使う
        self.docid_token_ids = self.tokenizer.encode(" A B C D E F G H I J K L M N O P Q R S T U V W X Y Z")
```

### compute_loss — ここが核心

```python
def compute_loss(self, model, inputs, return_outputs=False):
    input_ids = inputs["input_ids"]          # (B, L)
    prompt_lengths = inputs["prompt_lengths"]  # (B,)
```

**Step 1: マスクをかける**

```python
if self.args.mask_strategy == 'default':
    # シンプルなランダムマスキング
    noisy_batch, masked_indices, p_mask = forward_process(...)
elif self.args.mask_strategy == 'ranking_aware':
    # 文書IDトークン(A〜Z)だけをマスク
    noisy_batch, masked_indices, p_mask = ranking_aware_forward_process(...)
```

**Step 2: 回答の長さを計算**

```python
answer_lengths = (input_ids.shape[1] - prompt_lengths).unsqueeze(1)
# shape: (B, 1) → 各サンプルの回答部分のトークン数
```

**Step 3: モデルで予測**

```python
outputs = model(input_ids=noisy_batch)
# → logits shape: (B, L, vocab_size)
```

**Step 4: loss 計算（ここが独特）**

```python
# 普通の CrossEntropyLoss:
#   loss = CE(logits[all], labels[all])

# 拡散モデルの loss:
#   マスクした位置だけの CE を、
#   マスク率と回答長で正規化する

# (a) マスクした位置だけの CE（正規化なし）
token_loss = F.cross_entropy(
    logits[masked_indices],         # mask された位置の予測
    input_ids[masked_indices],      # 元の正解トークン
    reduction='none'                # 各トークンごとの loss を保持
)
# → shape: (total_masked_tokens,)

# (b) マスク確率で割る（低確率でマスクされたトークンは loss を重視）
token_loss /= p_mask[masked_indices]

# (c) 回答長で割り、バッチ平均を取る
ce_loss = torch.sum(
    token_loss / answer_lengths[masked_indices]
) / input_ids.shape[0]
```

### なぜこの loss 設計か

| 正規化 | 理由 |
|--------|------|
| **`/ p_mask`** | t が小さい（マスクが少ない）と loss も小さくなる。マスク率で割ることで t によらない loss にする |
| **`/ answer_length`** | 回答が長いサンプルほど loss が大きくなるのを防ぐ |
| **`/ B`** | バッチサイズが変わっても loss の規模が変わらないように |

### 図解：loss 計算の流れ

```
noisy_batch: [101, 205, MASK, MASK, 415]  (B=1, L=5, prompt_length=2)
                         ↑ここを予測
model(noisy_batch)
  → logits: tensor(1, 5, 126464)

logits[masked_indices]: tensor(2, 126464)  ← MASKされた2トークン分だけ
input_ids[masked_indices]: tensor([310, 320])  ← 正解

CE 各トークン: [1.2, 0.8]           ← reduction='none'
/ p_mask:      [1.2/0.5, 0.8/0.3]  = [2.4, 2.67]
/ answer_length: [2.4/3, 2.67/3]    = [0.8, 0.89]
sum / B:       (0.8 + 0.89) / 1    = 1.69  ← 最終 loss
```

---

## なぜ SFTTrainer を自前実装するのか

`transformers.Trainer` のデフォルトの `compute_loss` は:

```python
# デフォルト:
loss = model(input_ids, labels=labels).loss  # 普通の CE
```

拡散モデルでは「入力にマスクをかけてからモデルに通し、マスクした位置だけ loss を計算する」という特殊な処理が必要。そのため `compute_loss` を override している。

---

## forward_process と ranking_aware_forward_process の違い

| | forward_process | ranking_aware_forward_process |
|---|---|---|
| **マスク対象** | 回答部分の全トークン | 回答部分のうち文書IDトークン(A-Z)だけ |
| **狙い** | 基本形 | 文書IDの予測に特化 |
| **使うconfig** | `mask_strategy: default` | `mask_strategy: ranking_aware` |

今回の写経では `forward_process`（シンプル版）のみ書けばよい。
