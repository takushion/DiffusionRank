# 01: prepare_data.py — データ準備

## このファイルの目的

DiffusionRank の SFT 訓練に使うデータセットを作成する。

HuggingFace にある RankZephyr 訓練データ（castorini/rank_zephyr_training_data）をダウンロードし、SFTDataset が読める JSONL 形式に変換する。

## データ形式の変換

### 元データ（RankZephyr, 各行）

```json
{
  "id": "identity_xxx",
  "conversations": [
    {"from": "system", "value": "You are RankLLM, an intelligent assistant..."},
    {"from": "human",  "value": "I will provide you with 20 passages, each indicated by a numerical identifier []. Rank the passages based on their relevance to the search query: {query}.\n\n[1] {doc1 text}\n[2] {doc2 text}\n...\n[20] {doc20 text}\nSearch Query: {query}\nRank the 20 passages above..."},
    {"from": "gpt",    "value": "[3] > [8] > [14] > ... > [19]"}
  ]
}
```

各サンプルは3ターンの会話（system / human / gpt）で構成される。

- **human**: クエリ + 文書リスト（数値ID `[1]`〜`[N]`）+ 指示
- **gpt**: 正解ランキング（`[3] > [8] > [14] > ...`）

### 変換後（DiffusionRank 用, 各行）

```json
{
  "query": "fibroblasts synthesize what of the ground substance",
  "document": ["doc1 text...", "doc2 text...", ..., "doc20 text..."],
  "ranking": [3, 8, 14, 15, 20, 17, 16, 6, 9, 2, 1, 13, 10, 18, 7, 11, 4, 5, 12, 19]
}
```

3つのフィールドに構造化する:
- `query`: 検索クエリ文字列（`"Search Query: "` の後ろから抽出）
- `document`: 文書テキストのリスト（`[N]` 行頭マーカーから抽出）
- `ranking`: 正解順位の数値リスト（1始まり、gpt から抽出）

## SFTDataset がこのデータをどう使うか

SFTDataset の内部:

```python
documents = raw_data[i]["document"]  # ← ["doc1", "doc2", ...]
messages = [
    {"role": "user", "content": prompt.format(
        query=raw_data[i]["query"],
        documents="Document A: doc1\nDocument B: doc2\n..."
    )},
    {"role": "assistant", "content": "Ranking (most to least relevant): C H A"}
]
```

ポイント:
- 数値ID `[1]`〜`[N]` → アルファベット `A/B/C...` に変換される
- ランキング `[3, 8, 1]` → `"C H A"`（`chr(ord('A') + ranking[i] - 1)`）
- モデルは「与えられた文書をアルファベットで順位付けする」タスクとして学習する

## なぜこの変換が必要か

| 理由 | 詳細 |
|------|------|
| **アルファベットID** | 拡散モデルがテキスト生成としてランキングを出力するため。数値より識別子として自然 |
| **JSONL 構造化** | 訓練時はクエリ・文書・ランキングを個別に扱う。生の会話形式だとパースが非効率 |
| **1ファイルに集約** | 訓練スクリプトが1つのデータパスを読むだけで済む |

## パースの注意点（確認済み）

実データで確認したフォーマット特性:

1. 全ドキュメントが **行頭 `[N]` 形式**（マルチライン文書なし）
2. `"Search Query: "` が文書リストの直前に再出現する → これでクエリを抽出
3. ランキング形式は常に `[N] > [M] > [L]`（カンマや改行なし）
4. 文書数は 2〜20 の範囲で可変
