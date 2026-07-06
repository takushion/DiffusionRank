import os
import sys
import pathlib
import yaml
import torch
from dataclasses import dataclass, field
from transformers import (
    TrainingArguments as HFTrainingArguments,
    AutoTokenizer,
    AutoConfig,
    AutoModel,
    set_seed,
)
from peft import LoraConfig, get_peft_model

from data import make_data_module
from trainer import SFTTrainer


@dataclass
class TrainingArguments(HFTrainingArguments):
    mask_strategy: str = "default"
    
def main():
    config_path = (
        sys.argv[1] if len(sys.argv) > 1 else "configs/llada1.5_vanilla_lora.yaml"
    )
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    set_seed(42)
    
    
    model_config = AutoConfig.from_pretrained(config["model"]["name"], trust_remote_code=True)
    model_config.rope_theta *= config["model"]["rope_scaling_factor"]
    
    model = AutoModel.from_pretrained(
        config["model"]["name"],
        config=model_config,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    model.model.set_activation_checkpointing("fine_grained")
    
    if config.get("lora", None):
        lora_config = LoraConfig(**config["lora"])
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        
    tokenizer = AutoTokenizer.from_pretrained(config["model"]["name"])
    
    data_module = make_data_module(tokenizer=tokenizer, data_path=config["data"]["data_path"])
    
    training_args = config["training"]
    trainer_class_SFT = SFTTrainer
    training_args = TrainingArguments(
        **training_args,
        ddp_find_unused_parameters=False,
        label_names=["input_ids", "prompt_lengths"]
    )
    
    trainer = trainer_class_SFT(
        model=model,
        processing_class=tokenizer,
        args=training_args,
        **data_module
    )
    
    if (list(pathlib.Path(training_args.output_dir).glob("checkpoint-*"))and not training_args.overwrite_output_dir):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    torch.cuda.synchronize()
    trainer.save_model()


if __name__ == "__main__":
    main()
        
    
