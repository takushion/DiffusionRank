import os
import sys
import pathlib
import yaml
import torch
import argparse
from dataclasses import dataclass, field
from transformers import (
    TrainingArguments as HFTrainingArguments,
    AutoTokenizer,
    AutoConfig,
    AutoModel,
    set_seed,
)
from peft import LoraConfig, get_peft_model
from transformers import BitsAndBytesConfig

from data import make_data_module
from trainer import SFTTrainer


@dataclass
class TrainingArguments(HFTrainingArguments):
    mask_strategy: str = "default"
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", nargs="?", default="configs/llada1.5_vanilla_lora.yaml")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--qlora", action="store_true", help="4-bit QLoRA for Colab")
    args, remaining = parser.parse_known_args()
    config_path = args.config

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    set_seed(42)
    
    
    model_config = AutoConfig.from_pretrained(config["model"]["name"], trust_remote_code=True)
    model_config.rope_theta *= config["model"]["rope_scaling_factor"]
    
    load_kwargs = dict(
        config=model_config,
        trust_remote_code=True,
    )
    if args.qlora:
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        load_kwargs["device_map"] = "auto"
        load_kwargs["low_cpu_mem_usage"] = True
    else:
        load_kwargs["torch_dtype"] = torch.bfloat16

    model = AutoModel.from_pretrained(
        config["model"]["name"],
        **load_kwargs,
    )
    # LLaDA custom fine-grained activation checkpointing (per-op within block)
    # LLaDAModelLM.model → LLaDAModel (inner), which has set_activation_checkpointing
    if hasattr(model, "model") and hasattr(model.model, "set_activation_checkpointing"):
        model.model.set_activation_checkpointing("fine_grained")
        print("LLaDA fine-grained activation checkpointing enabled")
    elif hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    
    if config.get("lora", None):
        lora_config = LoraConfig(**config["lora"])
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        
    tokenizer = AutoTokenizer.from_pretrained(config["model"]["name"])
    
    data_module = make_data_module(tokenizer=tokenizer, data_path=config["data"]["data_path"])
    
    training_args = config["training"]
    if args.output_dir is not None:
        training_args["output_dir"] = args.output_dir

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

    # マージして保存（eval用）
    merged = model.merge_and_unload()
    merged.save_pretrained(training_args.output_dir + "_merged")
    tokenizer.save_pretrained(training_args.output_dir + "_merged")


if __name__ == "__main__":
    main()
        
    
