import os 
import sys 
from typing import Tuple, List, Dict 
from argparse import Namespace, ArgumentParser
import logging

import torch  
from torch import nn, Tensor 
from torch.nn import functional as F 
from torch.optim import AdamW
from peft import (
    LoraConfig,
    get_peft_model
)
import wandb
from transformers import (
    Trainer, 
    TrainingArguments, 
    AutoModelForCausalLM, 
    AutoTokenizer, 
)
from accelerate import Accelerator

from data.tasks import TASK_MAP
from data.collator import (QACollator, LMCollator)

if Accelerator().is_main_process:
    wandb.init()


# if Accelerator().is_main_process: 
#     import debugpy
#     debugpy.listen(("0.0.0.0", 5678))
#     print("Waiting for debugger attach...")
#     debugpy.wait_for_client() 


if __name__ == "__main__": 

    parser = ArgumentParser() 
    parser.add_argument("--model", type=str)
    parser.add_argument("--task", type=str, choices=["gsm", "pile"]) 
    parser.add_argument("--output_dir", type=str, default="runs/")
    parser.add_argument("--use_lora", action="store_true")
    parser.add_argument("--eval_every_steps",type=int, default=1000)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--weight_decay", type=float, default=0.01)

    args = parser.parse_args()

    if args.task == "gsm": 
        task_kwargs = {
            "list_splits": ["train", "test"], 
            "hold_out_validation": False
        }
    if args.task == "pile": 
        task_kwargs = {
            "list_splits": ["train", "test"]
        }
    else: 
        raise NotImplementedError() 

    datasets = TASK_MAP(args.task)(**task_kwargs)
    logging.info(f"dataset splits loaded: {datasets.get_splits()}")

    model = AutoModelForCausalLM.from_pretrained(args.model)

    if args.use_lora:
        peft_config = LoraConfig(
            r=16,                  
            lora_alpha=32,         
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"], 
            lora_dropout=0.0,
            bias="none",
            task_type="CAUSAL_LM"
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()


    # we shall use the instruct model's tokenizers, as they have predefined chat templates
    if "llama" in args.model.lower():
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
    if "pythia" in args.model.lower(): 
        tokenizer = AutoTokenizer.from_pretrained(args.model)
    else: 
        raise NotImplementedError()

    if args.task in ["gsm"]:
        collator = QACollator(tokenizer)
    elif args.task in ["pile"]: 
        collator = LMCollator(tokenizer)

    train_args = TrainingArguments(
        output_dir=args.output_dir,
        do_train=True,
        do_eval=True,
        do_predict=True,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=2,
        num_train_epochs=-1 if args.task in ["pile"] else 3,
        warmup_steps=0,
        gradient_accumulation_steps=8,
        bf16=True,
        bf16_full_eval=True,
        logging_strategy="steps", 
        logging_steps=10,
        report_to="wandb",
        eval_strategy="steps",
        eval_steps=args.eval_every_steps,
        save_strategy="steps",
        save_steps=args.eval_every_steps,
        max_steps= 42000 if args.task == "pile" else None,  
        metric_for_best_model="loss",
        greater_is_better=False,
        resume_from_checkpoint=None, 
        remove_unused_columns=False,
        dataloader_num_workers=4, 
        accelerator_config={"dispatch_batches": False, "split_batches": False}
    )

    optim_cls = AdamW 
    optim_kwargs = {
        "lr": args.lr, 
        "betas": (args.beta1, args.beta2), 
        "weight_decay": args.weight_decay, 
    }

    Accelerator().wait_for_everyone()

    trainer = Trainer(
        model=model,
        args=train_args,
        data_collator=collator,
        train_dataset=datasets["train"],
        eval_dataset=datasets["test"],
        processing_class=tokenizer,
        optimizer_cls_and_kwargs=(optim_cls, optim_kwargs)
    )
    trainer.train()