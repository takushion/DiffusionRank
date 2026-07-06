import re
import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from transformers import AutoTokenizer, AutoConfig, AutoModel


class LladaForEval:
    def __init__(
        self,
        model_path: str,
        rope_scaling_factor: float = 1.0,
        mask_id=126336,
        eos_id=126081,
        device="cuda",
        **kwargs,
    ):
        hf_kwargs = {"trust_remote_code": True, **kwargs}
        config = AutoConfig.from_pretrained("GSAI-ML/LLaDA-1.5", **hf_kwargs)
        config.rope_theta = config.rope_theta * rope_scaling_factor
        self.model = AutoModel.from_pretrained(
            model_path,
            config=config,
            torch_dtype=torch.bfloat16,
            **hf_kwargs,
        )
        self.model.eval()

        self.device = torch.device(device)
        self.model = self.model.to(self.device)

        self.mask_id = mask_id
        self.eos_id = eos_id
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        
    @torch.no_grad()
    def get_logits(self, batch, prompt_index=None, cfg=0.0):
        if cfg > 0.0:
            assert len(prompt_index) == batch.shape[1]
            prompt_index = prompt_index.unsqueeze(0).repeat(batch.shape[0], 1)
            un_batch = batch.clone()
            un_batch[prompt_index] = self.mask_id
            batch = torch.cat([batch, un_batch])

        logits = self.model(batch).logits

        if cfg > 0.0:
            logits, un_logits = torch.chunk(logits, 2, dim=0)
            logits = un_logits + (cfg + 1) * (logits - un_logits)
        return logits[:, : batch.shape[1]]

class WrapperBase:
    def __init__(self, model: LladaForEval, **kwargs):
        self.model = model

    def __call__(self, *args, **kwargs):
        raise NotImplementedError

class PermutationListwiseWrapper(WrapperBase):
    _prompt_template = """Given a query and {num} documents indicated by a character identifier, rank the documents from most relevant to least relevant to the query.

You should output a ranking using the document identifier, from most relevant to least relevant, separated by spaces.

Query: {query}

Documents:
{documents}
"""

    def __init__(
        self,
        model,
        num_steps: int = 1,
        inference_strategy: str = "assignment",
        **kwargs,
    ):
        super().__init__(model, **kwargs)
        self.num_steps = int(num_steps)
        assert inference_strategy in ["assignment", "sampling"]
        self.inference_strategy = inference_strategy
        
    def __call__(self, query: str, docs: list[str], **kwargs):
        n = len(docs)
        if n <= 0:
            return [], {
                "inputs": {"query": query, "documents": docs},
                "output": {"final_ranking": [], "steps": []},
            }

        mask_tok = "<|mdm_mask|>"

        index_strs = [f" {chr(i + ord('A'))}" for i in range(n)]
        index_token_ids = self.model.tokenizer.encode(
            "".join(index_strs), add_special_tokens=False
        )
        id_to_idx = {s: i for i, s in enumerate(index_strs)}

        docs_block = "\n".join(
            [f"Document {chr(i + ord('A'))}: {doc}" for i, doc in enumerate(docs)]
        )

        ranking_tokens_str = [mask_tok] * n
        all_steps_info = []
        
        for step in range(self.num_steps):
            frac = float(step + 1) / float(self.num_steps)
            target_filled = min(n, max(1, int(round(frac * n))))

            current_filled_positions = [
                i for i, tok in enumerate(ranking_tokens_str) if tok != mask_tok
            ]
            current_filled = len(current_filled_positions)

            num_to_new_fill = max(0, target_filled - current_filled)

            ranking_line = "Ranking (most to least relevant):" + "".join(
                ranking_tokens_str
            )

            messages = [
                {
                    "role": "user",
                    "content": self._prompt_template.format(
                        num=n, query=query, documents=docs_block
                    ),
                },
                {"role": "assistant", "content": ranking_line},
            ]

            input_ids = self.model.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=False,
                return_tensors="pt",
            ).to(self.model.device)

            mask_batch_idx, mask_token_idx = (input_ids == self.model.mask_id).nonzero(
                as_tuple=True
            )

            if mask_token_idx.numel() == 0:
                break

            logits = self.model.get_logits(input_ids)
            masked_logits = logits[mask_batch_idx, mask_token_idx, :]
            probs = torch.softmax(masked_logits, dim=-1)

            S_partial = probs[:, index_token_ids]  # [M, n]
            S_partial_np = S_partial.float().detach().cpu().numpy()
            
            used_docs = set()
            for tok in ranking_tokens_str:
                if tok != mask_tok and tok in id_to_idx:
                    used_docs.add(id_to_idx[tok])

            mask_positions = [
                i for i, tok in enumerate(ranking_tokens_str) if tok == mask_tok
            ]
            M = len(mask_positions)
            assert M == S_partial_np.shape[0]
            avail_docs = [j for j in range(n) if j not in used_docs]
            A = len(avail_docs)

            new_fills_this_step = []

            if num_to_new_fill > 0 and A > 0 and M > 0:
                S_avail = S_partial_np[:, avail_docs]  # [M, A]

                if self.inference_strategy == "assignment" and S_avail.size > 0:
                    cost = -np.log(np.clip(S_avail, 1e-12, 1.0))
                    row_ind, col_ind = linear_sum_assignment(cost)
                    assign_probs = S_avail[row_ind, col_ind]

                    order = np.argsort(-assign_probs)
                    k = min(num_to_new_fill, len(order))

                    for t in order[:k]:
                        m = int(row_ind[t])
                        a = int(col_ind[t])
                        doc_j = int(avail_docs[a])
                        prob_ = float(assign_probs[t])

                        rank_pos = mask_positions[m]
                        ranking_tokens_str[rank_pos] = index_strs[doc_j]
                        used_docs.add(doc_j)
                        new_fills_this_step.append((rank_pos, doc_j, prob_))

                else:
                    pair_list = []
                    for m in range(M):
                        for a, doc_j in enumerate(avail_docs):
                            pair_list.append((S_avail[m, a], m, doc_j))
                    pair_list.sort(key=lambda x: x[0], reverse=True)

                    assigned_m = set()
                    newly_used = set()
                    filled = 0
                    for score, m, doc_j in pair_list:
                        if filled >= num_to_new_fill:
                            break
                        if m in assigned_m or doc_j in newly_used or doc_j in used_docs:
                            continue
                        assigned_m.add(m)
                        newly_used.add(doc_j)

                        rank_pos = mask_positions[m]
                        ranking_tokens_str[rank_pos] = index_strs[doc_j]
                        used_docs.add(doc_j)
                        new_fills_this_step.append((rank_pos, doc_j, float(score)))
                        filled += 1

            all_steps_info.append(
                {
                    "step": step,
                    "ranking_tokens_str": ranking_tokens_str.copy(),
                    "target_filled": target_filled,
                    "num_to_new_fill": num_to_new_fill,
                    "new_fills_this_step": new_fills_this_step,
                    "used_docs_after_step": sorted(list(used_docs)),
                }
            )

            if all(tok != mask_tok for tok in ranking_tokens_str):
                break
        
        final_ranking = []
        for tok in ranking_tokens_str:
            if tok in id_to_idx:
                final_ranking.append(id_to_idx[tok])
            else:
                used = set(final_ranking)
                candidates = [i for i in range(n) if i not in used]
                final_ranking.append(candidates[0] if candidates else 0)

        seen = set()
        cleaned = []
        for idx in final_ranking:
            if 0 <= idx < n and idx not in seen:
                seen.add(idx)
                cleaned.append(idx)
        for i in range(n):
            if i not in seen:
                cleaned.append(i)
        final_ranking = cleaned

        outputs = {
            "inputs": {
                "query": query,
                "documents": docs,
                "prompts": self.model.tokenizer.apply_chat_template(
                    messages, add_generation_prompt=False, tokenize=False
                ),
                "num_steps": self.num_steps,
            },
            "output": {
                "final_ranking": final_ranking,
                "final_ranking_identifiers": [index_strs[i] for i in final_ranking],
                "steps": all_steps_info,
            },
        }

        return final_ranking, outputs
            
            
            
        
        
