# 02: data.py — データセットとコレータ

## このファイルの役割

```
JSONL (生データ)
    ↓ SFTDataset.__getitem__
各サンプル → {"input_ids": tensor(L,), "prompt_lengths": int}
    ↓ SFTDataCollator.__call__
バッチ    → {"input_ids": tensor(B, max_L), "prompt_lengths": tensor(B,)}
    ↓ Trainer へ
```

2つのクラスから構成される:

1. **SFTDataset**: 1サンプルずつ読み込み、プロンプトを組み立て、トークン化する
2. **SFTDataCollator**: バッチ内の異なる長さのサンプルをパディングして1つのテンソルに揃える

---

## SFTDataset の内部構造

### コンストラクタ (`__init__`)

```python
def __init__(self, data_path: str, tokenizer):
    with open(data_path) as f:
        self.raw_data = [json.loads(line) for line in f]
```

JSONL を全行読み込む。各行は事前に prepare_data.py で作った `{query, document, ranking}` 形式。

### 1サンプルの組み立て (`__getitem__`)

```python
documents = self.raw_data[i]["document"]  # ["doc1", "doc2", ...]
messages = [
    {
        "role": "user",
        "content": "Given a query and {num} documents...\n\n"
                    "Query: {query}\n\n"
                    "Documents:\n"
                    "Document A: doc1\nDocument B: doc2\n..."
    },
    {
        "role": "assistant",
        "content": "Ranking (most to least relevant): C H A"
    }
]
```

**ポイント**:

- 文書IDは `chr(ord('A') + i)` でアルファベットに変換
  - i=0 → 'A', i=1 → 'B', i=2 → 'C'
- ランキングの数値も同様にアルファベットに変換
  - `chr(ord('A') + x - 1)` （x は 1始まりのランキング値）
  - x=3 → 'C', x=8 → 'H', x=1 → 'A'

### トークン化（具体例で見る）

```python
# 例: 3文書の場合
documents = ["DocA本文...", "DocB本文...", "DocC本文..."]
# messages はこうなる:
#   user: "Given a query and 3 documents...\n\nQuery: 質問文\n\nDocuments:\nDocument A: DocA本文...\nDocument B: DocB本文...\nDocument C: DocC本文..."
#   assistant: "Ranking (most to least relevant): C A B"

input_ids = tokenizer.apply_chat_template(
    messages,
    truncation=True, max_length=8192,
    add_generation_prompt=False,  # ← assistant の回答も含める（訓練用）
    return_tensors="pt",
    return_dict=True
).input_ids[0]
```

`apply_chat_template` の戻り値は `{"input_ids": tensor([[1, 2, ..., N]]), "attention_mask": tensor([[1, 1, ..., 1]])}`。

`.input_ids[0]` で `shape=(1, L)` → `(L,)` に flatten する。Dataset は1サンプルを返すだけなので、バッチ次元は不要。DataCollator が後で複数サンプルを束ねる。

### prompt_lengths の意味

```python
prompt_length = len(tokenizer.apply_chat_template(
    messages[:-1],               # ← user メッセージだけ
    add_generation_prompt=True,  # ← "assistant" の開始までで止める
    return_dict=True
).input_ids)
```

```
                    prompt_length
├───────────────────────┼─────────────────────────┤
  <|user|>\n...\n<|assistant|>\n    C A B ...<|end|>
   ↑ ここにはマスクしない        ↑ ここだけマスクする
```

**なぜ必要か**: 拡散モデルの forward process では「アシスタントが生成すべき回答部分」だけをランダムにマスクする。プロンプト部分（ユーザーの指示・クエリ・文書）は常に見えたままにする。その境界を知るのが `prompt_length`。

もし `prompt_length` がなければ「クエリ自体もランダムに消される」ことになり、モデルは何をランキングすべきかわからなくなる。

---

## SFTDataCollator — なぜ必要か

### 問題

Dataset は1サンプルずつ返す:

```python
dataset[0] → {"input_ids": tensor([101, 205, 310]),          "prompt_lengths": 2}
dataset[1] → {"input_ids": tensor([101, 305, 410, 512, 613]), "prompt_lengths": 3}
dataset[2] → {"input_ids": tensor([101, 205]),                "prompt_lengths": 1}
#            ↑ 長さがバラバラ！
```

モデルはバッチを1枚のテンソル `(B, L)` で受け取る。長さを揃える必要がある。

### Step 1: キーごとに値を集める

```python
input_ids = [tensor([101, 205, 310]),
             tensor([101, 305, 410, 512, 613]),
             tensor([101, 205])]

prompt_lengths = [2, 3, 1]
```

このワンライナー:

```python
input_ids, prompt_lengths = tuple(
    [instance[key] for instance in instances]
    for key in ("input_ids", "prompt_lengths")
)
```

は以下と同じ:

```python
items = []
for key in ("input_ids", "prompt_lengths"):
    values = [inst[key] for inst in instances]
    items.append(values)
input_ids, prompt_lengths = tuple(items)
# input_ids = [tensor([101, 205, 310]), tensor([101, 305, 410, 512, 613]), tensor([101, 205])]
# prompt_lengths = [2, 3, 1]
```

### Step 2: パディング（長さを揃える）

```python
torch.nn.utils.rnn.pad_sequence(
    input_ids,                           # 長さバラバラのテンソルリスト
    batch_first=True,                    # 出力を (B, L) に
    padding_value=self.tokenizer.pad_token_id  # 埋める値（足りない部分をこれで埋める）
)
```

結果:

```
input_ids (3, 5):
[[101, 205, 310, pad, pad],     ← もともと3トークン、残り2つを pad で埋めた
 [101, 305, 410, 512, 613],     ← これが最長、そのまま
 [101, 205, pad,  pad, pad]]    ← もともと2トークン、残り3つを pad で埋めた
```

### Step 3: prompt_lengths をテンソルに

```python
prompt_lengths = torch.tensor([2, 3, 1], dtype=torch.long)
# shape: (3,)
```

### 完成品

```python
# Trainer の内部で DataLoader がバッチを作るたびに呼ばれる:
batch = collator([dataset[0], dataset[1], dataset[2]])

# batch = {
#     "input_ids": tensor([[101, 205, 310, pad, pad],
#                          [101, 305, 410, 512, 613],
#                          [101, 205, pad,  pad, pad]]),   # shape: (3, 5)
#     "prompt_lengths": tensor([2, 3, 1]),                   # shape: (3,)
# }

# この batch が SFTTrainer.compute_loss() に渡される
```

---

## 全体のデータの流れ（具体例）

```
JSONL 1行目: {"query": "機械学習とは", "document": ["DocA...", "DocB...", "DocC..."], "ranking": [3, 1, 2]}

SFTDataset.__getitem__(0):
  → messages = [
      {"role": "user", "content": "Given a query and 3 documents...\n\nQuery: 機械学習とは\n\nDocuments:\nDocument A: DocA...\nDocument B: DocB...\nDocument C: DocC..."},
      {"role": "assistant", "content": "Ranking (most to least relevant): C A B"}
    ]
  → input_ids = [user_token, ..., assistant_token, ..., C, A, B, end_token]  (L=87)
  → prompt_length = 70

SFTDataset.__getitem__(1):
  → input_ids = [...]  (L=120)
  → prompt_length = 95

SFTDataCollator([dataset[0], dataset[1]]):
  → input_ids = tensor(2, 120)  ← 長い方(120)に合わせてパディング
  → prompt_lengths = tensor([70, 95])
```

## 出力形式まとめ

| フィールド | 型 | shape | 説明 |
|-----------|-----|-------|------|
| `input_ids` | `torch.LongTensor` | `(B, max_L)` | パディング済みトークン列 |
| `prompt_lengths` | `torch.LongTensor` | `(B,)` | 各サンプルのプロンプト長（マスクしない部分） |

このバッチが `SFTTrainer.compute_loss()` に渡され、forward_process → model → loss と流れる。
