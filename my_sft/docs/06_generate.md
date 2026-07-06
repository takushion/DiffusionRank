# 06: generate.py — 逆拡散過程によるテキスト生成

## このファイルの役割

学習済みMDMからランキングテキストを生成する逆拡散過程（reverse process）を実装する。訓練時は forward process（テキスト→ノイズ）を学習し、推論時は **完全マスクから徐々に確定トークンへ遷移** する。

## 全体の流れ

```
完全マスク（gen_length分）+ プロンプト
  → ブロック分割（block_lengthずつ）
    → 各ブロック内で steps 回の reverse step
      → get_num_transfer_tokens で各stepの遷移数を計算
      → model(x) で logits を取得
      → get_transfer_index でどのマスクを確定するか選択
      → x[transfer_index] = x0[transfer_index] で確定
```

## 関数詳細

### `add_gumbel_noise(logits, temperature)`

Gumbel-max トリックでカテゴリカル分布からサンプリングする。

```python
def add_gumbel_noise(logits, temperature):
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (-torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise
```

- `temperature=0` → 決定論的（argmax）
- `temperature>0` → ノイズ加えて確率的サンプリング
- LLaDA の知見（arXiv:2409.02908）に基づき float64 を使用

### `get_num_transfer_tokens(mask_index, steps)`

各 reverse step で何トークンを確定させるか計算する。

```python
def get_num_transfer_tokens(mask_index, steps):
    mask_num = mask_index.sum(dim=1, keepdim=True)
    base = mask_num // steps
    remainder = mask_num % steps
    num_transfer_tokens = ... + base
    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, :remainder[i]] += 1
```

線形ノイズスケジュールのもと各stepの期待遷移数は一定なので、`mask_num / steps` を均等配分する。

### 3つのgenerate関数

| 関数 | KV cache | 特徴 |
|------|----------|------|
| `generate` | なし（毎回フル計算） | ベースライン、計算量大 |
| `generate_with_prefix_cache` | prompt部のみcache | ブロック内は毎回forward |
| `generate_with_dual_cache` | あり + replace_position | ブロック内もcache再利用、最速 |

### `generate`（基本形）

ブロック単位の reverse diffusion:

```python
for num_block in range(num_blocks):
    block_mask = x[block_start:block_end] == mask_id
    num_transfer_tokens = get_num_transfer_tokens(block_mask, steps)
    for i in range(steps):
        logits = model(x).logits
        mask_index[:, 次ブロック以降] = 0   # 未処理ブロックは触らない
        x0, transfer_index = get_transfer_index(logits, ...)
        x[transfer_index] = x0[transfer_index]
        if ブロック内マスクが0になったら break
```

gen_length 全体を一度に生成せず block_length で区切ることで、各ブロック内で **部分的な reverse process** を完結させる。

### `generate_with_prefix_cache`

初回 forward で KV cache を取得し、以降は cache を再利用する。

```python
output = model(x, use_cache=True)
past_key_values = output.past_key_values
# cache を current_block_start までトリミング
new_past_key_values[i] += (past_key_values[i][j][:, :, :current_block_start],)
# reverse step では generation部のみforward
logits = model(x[:, current_block_start:], past_key_values=past_key_values, use_cache=True).logits
```

### `generate_with_dual_cache`

`replace_position` で変更位置のみ cache を更新する。

```python
replace_position = torch.zeros_like(x, dtype=torch.bool)
replace_position[:, current_block_start:current_block_end] = 1
logits = model(x[:, current_block_start:current_block_end],
               past_key_values=past_key_values, use_cache=True,
               replace_position=replace_position).logits
```

### `get_transfer_index(logits, temperature, remasking, mask_index, x, num_transfer_tokens, threshold)`

**どのマスクを確定させるか** を決定する核心関数。

```python
x0 = argmax(add_gumbel_noise(logits, temperature))  # 仮の次のトークン
x0 = where(mask_index, x0, x)                       # 確定済みは保持
# confidence（確信度）の計算
if remasking == 'low_confidence':
    confidence = softmax(logits)[x0]                  # 予測確率
elif remasking == 'random':
    confidence = random()                             # ランダム
confidence = where(mask_index, confidence, -inf)      # 既存トークンは除外
# top-k を確定
for j in range(batch):
    _, select_index = topk(confidence[j], k=num_transfer_tokens[j])
    transfer_index[j, select_index] = True
    if threshold is not None:  # 低確信度はスキップ
        for k in range(1, num_transfer_tokens[j]):
            if confidence[j, select_index[k]] < threshold:
                transfer_index[j, select_index[k]] = False
```

**remasking戦略**:
- `low_confidence`: 最も確信しているトークンから確定（標準）
- `random`: ランダムに確定（探索的生成）

### `get_transfer_index_dynamic`

動的信頼度閾値を導入したバリアント。

```python
threshs = [1 - factor/(n+1) for n in range(num_transfer_tokens)]
threshs[0] = -1  # 最低1つは確定
for top_i in range(len(threshs)):
    if sorted_confidence[top_i] < threshs[top_i]:
        break
_, select_index = topk(confidence[j], k=top_i)
```

`factor` 大 → 早期確定が増える（高速だが品質低下リスク）。論文 Figure 4 の動的スケジューリングに相当。

## SFTとの関係

評価時に使用。`PermutationListwiseWrapper`（後の eval_utils.py）は reverse diffusion を1 step だけ実行し、各 position のスコア分布を取得する。
