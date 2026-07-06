import itertools
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import Trainer


def forward_process(
    input_ids: torch.Tensor,
    prompt_lengths: torch.Tensor,
    mask_token_id: int = 126336,
    eps: float = 1e-3,
):
    B, L = input_ids.shape
    device = input_ids.device

    t = torch.rand(B, device=device)
    p_mask_vals = (1 - eps) * t + eps
    rand_prob = torch.rand(B, L, device=device)
    mask_cond = rand_prob < p_mask_vals.unsqueeze(1)

    pos = torch.arange(L, device=device).unsqueeze(0).expand(B, -1)
    resp_mask = pos >= prompt_lengths.unsqueeze(1)
    mask_cond &= resp_mask

    noisy_batch = torch.where(mask_cond, mask_token_id, input_ids)
    p_mask_batch = torch.where(mask_cond, p_mask_vals.unsqueeze(1), 1.0)
    mask_idx_batch = mask_cond

    return noisy_batch, mask_idx_batch, p_mask_batch

def ranking_aware_forward_process(
    input_ids: torch.Tensor,
    prompt_lengths: torch.Tensor,
    docid_token_ids: list[int],
    mask_token_id: int = 126336,
    eps: float = 1e-3,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

    B, L = input_ids.shape
    device = input_ids.device

    t = torch.rand(B, device=device)
    p_mask_vals = (1 - eps) * t + eps
    rand_prob = torch.rand(B, L, device=device)
    mask_cond = rand_prob < p_mask_vals.unsqueeze(1)

    pos = torch.arange(L, device=device).unsqueeze(0).expand(B, -1)
    resp_mask = pos >= prompt_lengths.unsqueeze(1)
    mask_cond &= resp_mask

    docid_mask = torch.zeros_like(input_ids, dtype=torch.bool, device=device)
    for docid_token_id in docid_token_ids:
        docid_mask |= (input_ids == docid_token_id)
    docid_mask &= resp_mask

    mask_cond &= docid_mask

    noisy_batch = torch.where(mask_cond, mask_token_id, input_ids)
    p_mask_batch = torch.where(mask_cond, p_mask_vals.unsqueeze(1), 1.0)
    mask_idx_batch = mask_cond

    return noisy_batch, mask_idx_batch, p_mask_batch

class SFTTrainer(Trainer):
    def __init__(self, mask_token_id=126336, **kwargs):
        super().__init__(**kwargs)
        self.mask_token_id = mask_token_id
        self.docid_token_ids = self.tokenizer.encode(
            " A B C D E F G H I J K L M N O P Q R S T U V W X Y Z"
        )
    
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        input_ids = inputs["input_ids"]
        prompt_lengths = inputs["prompt_lengths"]

        if self.args.mask_strategy == "default":
            noisy_batch, masked_indices, p_mask = forward_process(
                input_ids,
                prompt_lengths,
                mask_token_id=self.mask_token_id,
            )
        elif self.args.mask_strategy == "ranking_aware":
            noisy_batch, masked_indices, p_mask = ranking_aware_forward_process(
                input_ids,
                prompt_lengths,
                docid_token_ids=self.docid_token_ids,
                mask_token_id=self.mask_token_id,
            )
        else:
            raise ValueError(f"Unknown mask_strategy: {self.args.mask_strategy}")
        
        # 損失関数の計算
        answer_lengths = (input_ids.shape[1] - prompt_lengths).unsqueeze(1)
        answer_lengths = answer_lengths.repeat(1, noisy_batch.shape[1])
        
        outputs = model(input_ids=noisy_batch)
        logits = outputs.logits
        
        token_loss = (
            F.cross_entropy(
                logits[masked_indices], input_ids[masked_indices], reduction="none")/ p_mask[masked_indices]
        )
        ce_loss = (
            torch.sum(token_loss / answer_lengths[masked_indices]) / input_ids.shape[0]
        )
        
        return (ce_loss, outputs) if return_outputs else ce_loss
        
        
        
        
        
        
        
