import os 
from typing import Tuple, List, Dict 
from copy import deepcopy

import torch
from torch import nn
from torch.nn import functional as F 
from torch.optim import AdamW
import torch.distributed.checkpoint as dist_cp
from torch.distributed.checkpoint.default_planner import DefaultLoadPlanner
from transformers import (
    AutoModelForCausalLM
)
from transformers.distributed.fsdp import get_fsdp_ckpt_kwargs, update_fsdp_plugin_peft
from peft import (
    LoraConfig,
    get_peft_model, 
    get_peft_model_state_dict, 
    set_peft_model_state_dict
)
from peft.utils.other import fsdp_auto_wrap_policy


from accelerate import (
    Accelerator, 
)
from accelerate.utils import load_fsdp_model


def peft_dcp_load_hook(models, input_dir):
    """
    Load PEFT adapter weights from a DCP checkpoint.

    The checkpoint from HF Trainer + FSDP + PEFT stores state_dict with keys like:
        model.base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight

    We need to match this exact structure for the DCP loader to work.
    """
    for model in models:
        # Generate a skeleton dictionary containing ONLY the adapter parameter keys.
        # This teaches the DCP loader exactly which keys to extract from the .distcp files.
        peft_state_dict = get_peft_model_state_dict(model)

        # Debug: print some keys to verify structure
        if accelerator.is_main_process:
            print(f"PEFT state_dict keys (first 3): {list(peft_state_dict.keys())[:3]}")

        # Wrap the target state dict matching PyTorch's distributed storage convention
        # The key "model" matches what HF Trainer uses for FSDP checkpoints
        state_dict_wrapper = {"model": peft_state_dict}

        # Read the distributed .distcp files in-place matching your active GPU ranks
        dist_cp.load(
            state_dict=state_dict_wrapper,
            storage_reader=dist_cp.FileSystemReader(input_dir),
            planner=DefaultLoadPlanner(),
        )

        if accelerator.is_main_process:
            print(f"After load - state_dict keys (first 3): {list(state_dict_wrapper['model'].keys())[:3]}")

        # Reinject the loaded adapter states back into the model
        set_peft_model_state_dict(model, state_dict_wrapper["model"])

    # CRITICAL: Empty the models list. This instructs Accelerator that the model
    # parameters are loaded via our custom hook, skipping its native loop which
    # expects full model parameters (not just LoRA adapters).
    while len(models) > 0:
        models.pop()


if __name__ == "__main__":

    accelerator = Accelerator()

    if accelerator.is_main_process:
        import debugpy
        debugpy.listen(("0.0.0.0", 5678))
        print("Waiting for debugger attach...")
        debugpy.wait_for_client()

    # Load the base model first (before PEFT wrapping)
    base_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B")

    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(base_model, peft_config)
    model.print_trainable_parameters()


    w_orig = deepcopy(get_peft_model_state_dict(model, adapter_name="default"))    

    load_fsdp_model(
        accelerator.state.fsdp_plugin, 
        accelerator, 
        model, 
        "/home/zonglin1/scratch/runs/dev/checkpoint-100", 
        **get_fsdp_ckpt_kwargs()
    )

    accelerator.wait_for_everyone()
    w_new = get_peft_model_state_dict(model, adapter_name="default")

    w_diff = {}
    for k in w_orig: 
        w_diff[k] = w_new[k] - w_orig[k] 

    accelerator.wait_for_everyone()
