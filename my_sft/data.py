from dataclasses import dataclass
import json
import math
from typing import Sequence, Any

import torch
from torch import Tensor
from torch.utils.data import Dataset
import transformers


class SFTDataset(Dataset):
    listwise_prompt = """Given a query and {num} documents indicated by a character identifier, rank the documents from most relevant to least relevant to the query.

You should output a ranking using the document identifier, from most relevant to least relevant, separated by spaces.

Query: {query}

Documents:
{documents}
"""

    def __init__(
        self,
        data_path: str,
        tokenizer: transformers.PreTrainedTokenizer,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        with open(data_path, "r") as f:
            self.raw_data = [json.loads(line) for line in f]
        print(f"Loaded {len(self.raw_data)} examples from {data_path}")
        
    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, i: int) -> dict[str, Any]:
        documents = self.raw_data[i]["document"]
        messages = [
            {
                "role": "user",
                "content": self.listwise_prompt.format(
                    num=len(documents),
                    query=self.raw_data[i]["query"],
                    documents="\n".join(
                        [
                            f"Document {chr(ord('A') + i)}: {doc}"
                            for i, doc in enumerate(documents)
                        ]
                    ),
                ),
            },
            {
                "role": "assistant",
                "content": "Ranking (most to least relevant):"
                + "".join(
                    [f" {chr(ord('A') + x - 1)}" for x in self.raw_data[i]["ranking"]]
                ),
            },
        ]
        input_ids = self.tokenizer.apply_chat_template(
            messages,
            truncation=True,
            max_length=8192,
            add_generation_prompt=False,
            return_tensors="pt",
        )[0]
        prompt_length = len(
            self.tokenizer.apply_chat_template(
                messages[:-1], add_generation_prompt=True
            )
        )
        ret = dict(
            input_ids=input_ids,
            prompt_lengths=prompt_length,
        )
        
        return ret
    
@dataclass
class SFTDataCollator:
    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[dict]) -> dict[str, Tensor]:
        input_ids, prompt_lengths = tuple(
            [instance[key] for instance in instances]
            for key in ("input_ids", "prompt_lengths")
        )
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        prompt_lengths = torch.tensor(prompt_lengths, dtype=torch.long)
        batch = dict(
            input_ids=input_ids,
            prompt_lengths=prompt_lengths,
        )
        return batch
        
def make_data_module(
    tokenizer: transformers.PreTrainedTokenizer,
    data_path: str,
) -> dict:
    train_dataset = SFTDataset(
            data_path,
            tokenizer=tokenizer,
        )
    data_collator = SFTDataCollator(tokenizer=tokenizer)

    return dict(
        train_dataset=train_dataset,
        data_collator=data_collator
    )
