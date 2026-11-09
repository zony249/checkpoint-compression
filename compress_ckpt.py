import os 
from typing import Tuple, List, Dict 
from copy import deepcopy
from argparse import ArgumentParser, Namespace
from glob import glob
from tqdm import tqdm
from collections import OrderedDict

import faiss
import torch
from torch import nn
from torch.nn import functional as F 
from torch.optim import AdamW
import torch.distributed.checkpoint as dist_cp
from torch.distributed.checkpoint.default_planner import DefaultLoadPlanner
from transformers import (
    AutoModelForCausalLM, 
    PreTrainedModel, 
)
from transformers.distributed.fsdp import get_fsdp_ckpt_kwargs, update_fsdp_plugin_peft
from peft import (
    LoraConfig,
    get_peft_model, 
    get_peft_model_state_dict, 
    set_peft_model_state_dict
)
from peft.utils.other import fsdp_auto_wrap_policy
from safetensors import safe_open 
from safetensors.torch import save_file


from accelerate import (
    Accelerator, 
)
from accelerate.utils import load_fsdp_model, load_fsdp_optimizer

CHECKPOINT_PREFIX="checkpoint-"
COMPRESSED_CKPT_PREFIX="cp-checkpoint-"

def prune(tensor, mag_thres = 0.5):
    mag_thres = min(2.5, mag_thres)
    mag_thres = tensor.abs().median() * mag_thres
    tensor = torch.where(tensor.abs() > mag_thres, tensor, 0)
    return tensor, (tensor == 0).sum()


def prune_optimizer(exp_tensor, exp_sq_tensor, tensor, j = 1):
    res_exp = torch.where((exp_tensor.abs()) > j * (exp_tensor.abs()).mean(), exp_tensor, 0)
    res_exp = torch.where(tensor == 0, 0, res_exp)
    res_exp_sq = torch.where((exp_tensor.abs()) > j * (exp_tensor.abs()).mean(), exp_sq_tensor, 0)
    res_exp_sq = torch.where(tensor == 0, 0, res_exp_sq)
    return res_exp, res_exp_sq, (res_exp == 0).sum()


def quant(tensor, bit_num = 4):
    assert 8 % bit_num == 0 and bit_num <= 8
    all_labels = []
    all_codebook = []
    knn_tensor = deepcopy(tensor)
    knn_tensor_shape = knn_tensor.shape
    knn_tensor_flt = knn_tensor.flatten()
    assert knn_tensor_flt.shape[0] % (8 // bit_num) == 0
    knn_tensor_nonzero = knn_tensor_flt[knn_tensor_flt != 0]
    # print(knn_tensor_nonzero.shape[0])
    if knn_tensor_nonzero.shape[0] <= 2 ** bit_num - 1:
        return tensor, None
    kmeans = faiss.Kmeans(1, 2 ** bit_num - 1, gpu=False)
    kmeans.train(knn_tensor_nonzero.view(-1, 1).numpy())
    codebook = kmeans.centroids
    labels = kmeans.index.search(knn_tensor_nonzero.view(-1, 1).numpy(), 1)[1]
    knn_tensor_flt[knn_tensor_flt.nonzero().squeeze()] = torch.Tensor(labels + 1).squeeze()
    knn_tensor_flt2 = knn_tensor_flt.view(-1, 8 // bit_num)
    knn_tensor_slim = torch.zeros_like(knn_tensor_flt2[:, 0])
    for i in range(8 // bit_num):
        knn_tensor_slim += knn_tensor_flt2[:, i] * ((2 ** bit_num) ** (8 // bit_num - i - 1))
    knn_tensor_flt = knn_tensor_slim.view(-1, knn_tensor_shape[-1] // (8 // bit_num))
    return knn_tensor_flt, codebook


def unquantize(codebook, indexes, bit_num):
    recover_indexes = torch.zeros(indexes.nelement(), 8 // bit_num)
    for i in range(8 // bit_num):
        recover_indexes[:, i] = indexes.flatten() // ((2 ** bit_num) ** (8 // bit_num - i - 1))
        indexes = indexes.fmod(((2 ** bit_num) ** (8 // bit_num - i - 1)))
    recover_tensor = torch.concatenate((torch.Tensor([0]), codebook.squeeze()), dim=0)[recover_indexes.flatten().squeeze().to(torch.int64)]
    return recover_tensor
 

def recon(checkpoint, ref_weights, compressed_optim, args):
    recon_dict = {}
    optimizer_dict = {}
    data_pt = torch.load(os.path.join(args.ref_checkpoint_path, "optimizer.pt"), map_location='cpu')
    for k, ko in zip(checkpoint, compressed_optim):
        ckpt = checkpoint[k]
        optim_k = ko
        ref_shape = ref_weights.get_tensor(k).shape
        if "codebook" in ckpt.keys():
            recover_weights = unquantize(ckpt["codebook"], ckpt["index"], args.quant_bits).view(ref_shape)
        else:
            recover_weights = ckpt["index"].view(ref_shape)
        if "v_codebook" in ckpt.keys():
            recover_opt_v = unquantize(ckpt["v_codebook"], ckpt["v_index"], args.quant_bits_opt).view(ref_shape)
        else:
            recover_opt_v = ckpt["v_index"].view(ref_shape)
        if "m_codebook" in ckpt.keys():
            recover_opt_m = unquantize(ckpt["m_codebook"], ckpt["m_index"], args.quant_bits_opt).view(ref_shape)
        else:
            recover_opt_m = ckpt["m_index"].view(ref_shape)
        print(k, recover_weights.shape, ref_weights.get_tensor(k).shape)
        recon_dict[k] = (recover_weights + ref_weights.get_tensor(k)).to(torch.float32)
        optim_k = ko
        data_pt['state'][optim_k]['exp_avg'] = recover_opt_v.to(torch.float32)
        data_pt['state'][optim_k]['exp_avg_sq'] = recover_opt_m.to(torch.float32)
    save_file(recon_dict, os.path.join(args.output_recon, "model.safetensors"), metadata = {"format": "pt"})
    torch.save(data_pt, os.path.join(args.output_recon, "optimizer.pt"))
    return recon_dict




def load_maybe_lora_fsdp_model(
    accelerator: Accelerator,
    base_model:PreTrainedModel,
    ckpt_path:str,
    use_lora:bool = False
) -> PreTrainedModel: 

    if use_lora:
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
    else: 
        model = base_model
    # w_orig = deepcopy(get_peft_model_state_dict(model, adapter_name="default"))    
    load_fsdp_model(
        accelerator.state.fsdp_plugin, 
        accelerator, 
        model, 
        ckpt_path, 
        **get_fsdp_ckpt_kwargs()
    )
    return model  



if __name__ == "__main__":

    parser = ArgumentParser() 
    parser.add_argument("--base_model", required=True, type=str)
    parser.add_argument("--use_lora", action="store_true")
    parser.add_argument("--checkpoints_dir", type=str, required=True)
    parser.add_argument("--output_compressed", type=str)
    parser.add_argument("--output_recon", type=str)
    parser.add_argument('--prune_alpha', type=float, default=5e-5)
    parser.add_argument('--prune_beta', type=float, default=2.0)
    parser.add_argument('--quant_bits', type=int, default=4)
    parser.add_argument('--quant_bits_opt', type=int, default=4)

    args = parser.parse_args()

    os.makedirs(args.output_compressed, exist_ok=True) 
    os.makedirs(args.output_recon, exist_ok=True)

    accelerator = Accelerator()

    # if accelerator.is_main_process:
    #     import debugpy
    #     debugpy.listen(("0.0.0.0", 5679))
    #     print("Waiting for debugger attach...")
    #     debugpy.wait_for_client()

    # Load the base model first (before PEFT wrapping)
    base_model = AutoModelForCausalLM.from_pretrained(args.base_model)

    checkpoint_path_prefix = os.path.join(args.checkpoints_dir, CHECKPOINT_PREFIX)
    ckpt_paths = glob(checkpoint_path_prefix + "*")
    number_extract = sorted([int(p.split(checkpoint_path_prefix)[-1]) for p in ckpt_paths])

    # collect trainable parameters
    if args.use_lora: 
        peft_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
            lora_dropout=0.0,
            bias="none",
            task_type="CAUSAL_LM"
        )
        lora_model = get_peft_model(base_model, peft_config)
        lora_model.print_trainable_parameters()
        trainable_params = [p for p in lora_model.parameters() if p.requires_grad]
    else: 
        trainable_params = base_model.parameters()

    optimizer = torch.optim.AdamW(trainable_params, lr=3e-4)

    """
    compressed_ckpt: {
        "weight_k" : {
            "index": tensor,        
            "codebook": tensor
        }
    }
    compressed_opt_states : {
        int: {
            "v_index": tensor, 
            "v_codebook": tensor, 
            "m_index": tensor, 
            "m_codebook: tensor
        }
    }
    """

    current_recon_ckpt = None
    # loop through directory of checkpoints
    # Assumes no lora for now #
    for i, n in enumerate(tqdm(number_extract, desc="Compressing")): 

        if i == 0: 
            prev_ckpt_path = args.base_model
            prev_ckpt = base_model 
            prev_recon_ckpt = base_model.state_dict() 
        else: 
            prev_ckpt_path = checkpoint_path_prefix + str(n) 
            prev_ckpt = load_maybe_lora_fsdp_model(accelerator, deepcopy(base_model), prev_ckpt_path, args.use_lora)
            prev_recon_ckpt = deepcopy(current_recon_ckpt)

        # load current ckpt and optimizer state
        current_ckpt_path = checkpoint_path_prefix + str(n) 
        current_ckpt = load_maybe_lora_fsdp_model(accelerator, deepcopy(base_model), current_ckpt_path, args.use_lora)
        load_fsdp_optimizer(accelerator.state.fsdp_plugin, accelerator, optimizer, base_model, current_ckpt_path, adapter_only=args.use_lora)
        current_opt_state = optimizer.state_dict()["state"] # odict with numeric keys


        prev_state_dict = prev_ckpt.state_dict()
        current_state_dict = current_ckpt.state_dict() # odict with string keys

        compressed_ckpts = OrderedDict() 
        compressed_opt_states = OrderedDict() 

        for kw, ko in zip(current_state_dict.keys(), current_opt_state.keys()): 

            weights = current_state_dict[kw] 
            optim_v = current_opt_state[ko]["exp_avg"] 
            optim_m = current_opt_state[ko]["exp_avg_sq"] 

            prev_weights = prev_state_dict[kw] 

            residual = weights - prev_weights 
            residual, remove = prune(residual, args.prune_alpha / optim_m.mean().sqrt().cpu())
            pruned_opt_v, pruned_opt_m, remove_opt = prune_optimizer(optim_v, optim_m, residual, args.prune_beta)
            residual_tensor_index, residual_tensor_codebook = quant(residual, args.quant_bits)

            opt_v_index, opt_v_codebook = quant(pruned_opt_v, args.quant_bits_opt)
            opt_m_index, opt_m_codebook = quant(pruned_opt_m, args.quant_bits_opt)

            compressed_ckpts[kw] = {}
            compressed_ckpts[kw]["index"] = torch.tensor(residual_tensor_index).to(torch.uint8)
            if residual_tensor_codebook is not None: 
                compressed_ckpts[kw]["codebook"] = torch.tensor(residual_tensor_codebook).to(torch.float16)

            compressed_opt_states[ko] = {} 
            compressed_opt_states[ko]["v_index"] = torch.tensor(opt_v_index).to(torch.uint8)
            compressed_opt_states[ko]["m_index"] = torch.tensor(opt_m_index).to(torch.uint8) 
            if opt_v_codebook is not None: 
                compressed_opt_states[ko]["v_codebook"] = torch.tensor(opt_v_codebook).to(torch.float16)
                compressed_opt_states[ko]["m_codebook"] = torch.tensor(opt_m_codebook).to(torch.float16)
            
            accelerator.wait_for_everyone()

        # save compressed diffs
        torch.save(compressed_ckpts, os.path.join(args.output_compressed, "compressed.pt"))

        current_recon_ckpt = recon(compressed_ckpts, prev_recon_ckpt, compressed_opt_states, args)

    # accelerator.wait_for_everyone()
    # w_new = get_peft_model_state_dict(model, adapter_name="default")

    # w_diff = {}
    # for k in w_orig: 
    #     w_diff[k] = w_new[k] - w_orig[k] 

    # accelerator.wait_for_everyone()
