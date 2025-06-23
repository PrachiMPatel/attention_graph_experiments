import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments, AutoConfig, Trainer, DataCollatorForSeq2Seq, HfArgumentParser
import datasets
from datasets import Dataset, load_dataset
import torch
import logging
import os
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training, set_peft_model_state_dict
import pandas as pd
import math
import bitsandbytes as bnb
import transformers
from typing import Dict
from typing import List, Optional
from accelerate import Accelerator
import numpy as np
import random
from datetime import datetime
import sys
from dataclasses import dataclass, field
from ast import literal_eval
import copy
from utils.prompt import generate_prompt

logger = logging.getLogger(__name__)

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The model checkpoint for weights initialization. Don't set if you want to train a model from scratch."
            )
        },
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    train_file: Optional[str] = field(default=None, metadata={"help": "The input training data file (a text file)."})
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    )
    validation_split_percentage: Optional[float] = field(
        default=0.1,
        metadata={
            "help": "The percentage of the train set used as validation set in case there's no validation split"
        },
    )


@dataclass
class OtherTrainingArguments:
    """
    Arguments pertaining to training that are not included in TrainingArguments.
    """

    per_device_backprop_batch_size: int = field(
        default=16, metadata={"help": "Batch size per GPU/TPU/MPS/NPU core/CPU for running the backward pass."}
    )
    wandb_project: Optional[str] = field(default=None, metadata={"help": "The wandb project to report to."})
    log_file: Optional[str] = field(default="train_seq2seq.log", metadata={"help": "The file to log to."})


def seed_all(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def tokenize(prompt):
    result = tokenizer(
        prompt,
        padding=False,
        return_tensors=None,
    )
    result["labels"] = result["input_ids"].copy()
    return result

def generate_and_tokenize_prompt(data_point):
    full_prompt = generate_prompt(data_point)
    tokenized_full_prompt = tokenize(full_prompt)

    data_point_copy = copy.deepcopy(data_point)
    data_point_copy.pop("response")
    user_prompt = generate_prompt(data_point_copy)
    tokenized_user_prompt = tokenize(user_prompt)

    user_prompt_len = len(tokenized_user_prompt["input_ids"])
    if tokenizer.add_eos_token:
        user_prompt_len -= 1
    tokenized_full_prompt["labels"] = [
        -100
    ] * user_prompt_len + tokenized_full_prompt["labels"][
        user_prompt_len:
    ]

    return tokenized_full_prompt

if __name__ == "__main__":
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, OtherTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, other_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, other_args, training_args = parser.parse_args_into_dataclasses()

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(other_args.log_file)],
    )

    transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    logger.info(f"Training/evaluation parameters {training_args}")

    seed_all(0)

    dataset = load_dataset('csv',data_files = data_args.train_file, split='train')
    split_dataset = dataset.train_test_split(test_size=data_args.validation_split_percentage)
    train_dataset = split_dataset["train"]
    eval_dataset = split_dataset["test"]

    gradient_accumulation_steps = other_args.per_device_backprop_batch_size // training_args.per_device_train_batch_size
    if '/' in training_args.run_name and training_args.run_name == training_args.output_dir:
        run_name = training_args.run_name.rstrip('/').split('/')[-1] + f"-{datetime.now().strftime('%Y-%m-%d-%H-%M')}"
    else:
        run_name = training_args.run_name

    os.environ["WANDB_PROJECT"]=other_args.wandb_project

    training_args = TrainingArguments(
            per_device_train_batch_size=training_args.per_device_train_batch_size,
            per_device_eval_batch_size=training_args.per_device_eval_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=training_args.warmup_steps,
            num_train_epochs=training_args.num_train_epochs,
            learning_rate=training_args.learning_rate,
            fp16=True,
            logging_steps=10,
            optim="adamw_torch",
            evaluation_strategy="steps", # if val_set_size > 0 else "no",
            save_strategy="steps",
            eval_steps=100,
            save_steps=100,
            output_dir=training_args.output_dir,
            # save_total_limit=3,
            load_best_model_at_end=False,
            logging_dir=training_args.logging_dir,
            # ddp_find_unused_parameters=False if ddp else None,
            group_by_length=True, # group sequences of roughly the same length together to speed up training
            report_to="wandb", # if use_wandb else "none",
            run_name=run_name, # if use_wandb else None,
            deepspeed=training_args.deepspeed,
            gradient_checkpointing=True,
        )

    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)

    tokenizer.add_eos_token = True
    tokenizer.pad_token_id = 0
    tokenizer.padding_side = "left"

    tokenized_train_dataset = train_dataset.map(generate_and_tokenize_prompt)
    tokenized_val_dataset = eval_dataset.map(generate_and_tokenize_prompt)

    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    else:
        def make_inputs_require_grad(module, input, output):
            output.requires_grad_(True)

        model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    config = LoraConfig(
        r=512,
        lora_alpha=256,
        target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
        "lm_head",
    ],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, config)

    model.train() 

    trainer = Trainer(
        model=model,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_val_dataset,
        args=training_args,
        data_collator=DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
    )

    trainer.train()