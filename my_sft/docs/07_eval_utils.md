# 07: eval_utils.py — 評価用ラッパー

## このファイルの役割

MDMに様々なインターフェースで評価を依頼するラッパー群を提供する。SFT評価で使うのは **`LladaForEval`**（モデルローダー＋推論エンジン）と **`PermutationListwiseWrapper`**（拡散1stepのスコア分布からランキングを構成）。

## LladaForEval

学習済みモデルをロードし、生成・尤度評価の窓口となる。

```python
class LladaForEval:
    def __init__(self, model_path, mask_id=126336, eos_id=126081,
                 max_length=4096, batch_size=8, mc_num=16,
                 steps=1024, gen_length=1024, block_length=1024,
                 remasking='low_confidence', threshold=None,
                 use_cache=False, dual_cache=False):
```

### 初期化
1. `AutoConfig.from_pretrained("GSAI-ML/LLaDA-1.5", trust_remote_code=True)` で config を取得（モデル自体のconfigはremoteから読む）
2. `rope_theta` を `rope_scaling_factor` で拡張（長い入力に対応）
3. `AutoModel.from_pretrained(model_path, ...)` で重みをロード（write-alongでは `modeling_llada.py` がないので `AutoModel` を使う）

### generate()

3つのgenerate関数を切り替える。

```python
@torch.no_grad()
def generate(self, input_ids, **kwargs):
    if self.dual_cache:
        output_ids, nfe, history = generate_with_dual_cache(...)
    elif self.use_cache:
        output_ids, nfe, history = generate_with_prefix_cache(...)
    else:
        output_ids, nfe, history = generate(...)
    return output_ids, nfe, history
```

### get_logits()

CFG（Classifier-Free Guidance）にも対応した logits 取得。

```python
@torch.no_grad()
def get_logits(self, batch, prompt_index=None, cfg=0.):
    if cfg > 0.:
        # プロンプト部をマスクした unconditional 版もまとめてforward
        un_batch = batch.clone()
        un_batch[prompt_index] = self.mask_id
        batch = torch.cat([batch, un_batch])
    logits = self.model(batch).logits
    if cfg > 0.:
        logits, un_logits = torch.chunk(logits, 2, dim=0)
        logits = un_logits + (cfg + 1) * (logits - un_logits)
    return logits
```

- CFG: `unconditional + (cfg + 1) * (conditional - unconditional)`
- `cfg=0` なら通常の条件付き logits

### loglikelihood()

与えられた prefix + target の対数尤度を、MC サンプリングで推定する。

```python
@torch.no_grad()
def loglikelihood(self, prefix, target):
    seq = concat([prefix, target])  # shape: [1, L]
    seq = seq.repeat(batch_size, 1)

    for _ in range(mc_num // batch_size):
        # forward process でランダムマスク
        perturbed_seq, p_mask = self._forward_process(seq, prompt_index)
        # logits 取得
        logits = self.get_logits(perturbed_seq, prompt_index)
        # mask 位置の cross-entropy を p_mask で重み付け
        loss = F.cross_entropy(logits[mask_indices], seq[mask_indices], reduction='none')
        loss = loss.sum() / batch_size
    return -sum(loss_acc) / len(loss_acc)
```

### _forward_process()

尤度評価用の特殊な forward process（訓練時とは異なる）。

```python
def _forward_process(self, batch, prompt_index):
    target_len = l - prompt_index.sum()
    k = randint(1, target_len + 1)
    # 各バッチ要素で異なるマスク数
    x = linspace(k, k + (b-1) * (target_len/b), steps=b).long()
    x = ((x - 1) % target_len) + 1
    # マスク位置をランダムに選択
    is_mask = indices < x.unsqueeze(1)  # 各行で異なるマスク数
    for i in range(b):
        is_mask[i] = is_mask[i][randperm(target_len)]  # シャッフル
```

**訓練時との違い**: 訓練では cosine schedule だが、評価時は **一様分布からのランダムマスク数 × ランダム位置**。これは MDM のロスが任意のマスク数・位置で定義できる性質を利用し、MC 積分で尤度を推定するため。

## WrapperBase

```python
class WrapperBase:
    def __init__(self, model: LladaForEval, **kwargs):
        self.model = model
    def __call__(self, *args, **kwargs):
        raise NotImplementedError
```

すべてのランキングラッパーの基底クラス。

## ListwiseGenerationWrapper

**全文生成型**: プロンプト→拡散生成→出力テキストをパース→ランキング。

```python
class ListwiseGenerationWrapper(WrapperBase):
    _prompt_template = """Given a query and {num} documents ...
    Query: {query}
    Documents:
    Document A: ...
    Document B: ...
    """
    def __call__(self, query, docs):
        messages = [{"role": "user", "content": template}]
        input_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
        output_ids, _, _ = model.generate(input_ids)
        output_text = tokenizer.decode(output_ids[0, input_ids.shape[1]:])
        ranking = parse_output(output_text, n)
        return ranking, outputs
```

**利点**: 自由文生成なので柔軟性が高い。**欠点**: 毎回フル生成が必要で遅い。

## PointwiseWrapper

**文書単位の判定**: 各文書を独立に評価して0/1のスコアを得る。

```python
class PointwiseWrapper(WrapperBase):
    def __call__(self, query, doc):
        messages = [{"role": "user", "content": prompt},
                    {"role": "assistant", "content": "<|mdm_mask|>"}]
        input_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=False)
        logits = model.get_logits(input_ids)
        yes_loc, no_loc = tokenizer.encode("1")[0], tokenizer.encode("0")[0]
        p_yes = softmax(logits)[yes_loc]
        p_no  = softmax(logits)[no_loc]
        score = p_yes / (p_yes + p_no)  # 1である確率
        return score, outputs
```

**特徴**: 各文書を独立に評価するので文書間の比較ができない。**SFTでは使わない**。

## LogitsListwiseWrapper

**logits一括評価型**: 全文書のスコアを1回のforwardで取得する。

```python
class LogitsListwiseWrapper(WrapperBase):
    def __call__(self, query, docs):
        # assistant側に各文書のマスク位置を用意
        assistant_content = "\n".join([f"Document {i}: <|mdm_mask|>" for i in range(n)])
        messages = [{"role": "user", "content": template},
                    {"role": "assistant", "content": assistant_content}]
        input_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=False)
        mask_batch_idx, mask_token_idx = (input_ids == mask_id).nonzero()
        logits = model.get_logits(input_ids)
        masked_logits = logits[mask_batch_idx, mask_token_idx, :]
        probs = softmax(masked_logits)
        p_yes = probs[:, yes_loc]
        p_no  = probs[:, no_loc]
        scores = p_yes / (p_yes + p_no)
        ranking = sorted(range(n), key=lambda i: scores[i], reverse=True)
        return ranking, outputs
```

**SFTでは使わない**（訓練対象外のため）。

## PermutationListwiseWrapper ★ SFT用

**段階的スコアリング型**: reverse diffusion を1 step だけ実行し、各 position のトークン確率分布から Hungarian matching で最適割り当てを計算する。

```python
class PermutationListwiseWrapper(WrapperBase):
    def __init__(self, model, num_steps=1, inference_strategy="assignment"):
        self.num_steps = num_steps
        self.inference_strategy = inference_strategy  # "assignment" or "sampling"
```

### アルゴリズム

```
入力: query, docs (n個)
出力: final_ranking (長さnの順列)

ranking_tokens = [<mask>, <mask>, ..., <mask>]  # n個

for step in range(num_steps):
    target_filled = round((step+1)/num_steps * n)  # 今回までに埋める数

    # プロンプト構築（現在の ranking_tokens 状態を assistant 側に）
    ranking_line = "Ranking: " + "".join(ranking_tokens)
    messages = [user: query+docs, assistant: ranking_line]

    # 1 step forward → logits
    input_ids = tokenizer.apply_chat_template(messages)
    mask_idx = (input_ids == mask_id).nonzero()
    logits = model.get_logits(input_ids)
    masked_logits = logits[mask_idx]
    probs = softmax(masked_logits)

    # 各マスク位置の「A」「B」「C」...の確率
    S_partial = probs[:, index_token_ids]  # shape: [M, n]

    # Hungarian 割り当て
    cost = -log(S_avail)                    # 最小化問題に変換
    row_ind, col_ind = linear_sum_assignment(cost)

    # 確率が高い順にソートし、必要な数だけ確定
    for t in order[:num_to_new_fill]:
        ranking_tokens[mask_positions[row_ind[t]]] = index_strs[col_ind[t]]
```

### Hungarian matching の意味

`S_partial[m, d]` = 「position m に document d が来る確率」と解釈する。各 position に異なる document を割り当てる **割り当て問題** なので、Hungarian アルゴリズムで最適な bijection を求める。

**コスト関数**: `cost = -log(p)` — 確率が高いほどコストが小さい。

### inference_strategy

- `"assignment"`（デフォルト）: Hungarian で最適割り当て。決定論的。
- `"sampling"`: greedy サンプリング。確率的（後処理のバリエーション用）。

### SFT評価での使い方

```python
model = LladaForEval(model_path="./my_sft/output/checkpoint-1000")
wrapper = PermutationListwiseWrapper(model, num_steps=1, inference_strategy="assignment")
ranking, outputs = wrapper(query="...", docs=[doc1, doc2, doc3])
```

`num_steps=1` の場合、1回の reverse step で全 position を一度に埋める（実質的に logits からの直接割り当て）。
