from typing import Dict, List, Tuple 
from copy import deepcopy 

import torch 
from torch.utils.data import Dataset, Subset 
from torch.nn.utils.rnn import pad_sequence
from transformers import (
    AutoTokenizer, 
    PreTrainedTokenizer, 
    BatchEncoding
)

class QACollator: 
    def __init__(self, tokenizer: PreTrainedTokenizer): 
        self.tokenizer = tokenizer
    def __call__(self, examples: Tuple) -> BatchEncoding: 

        batch_input_ids = []
        batch_labels = []
        batch_attn_mask = []
        for ex in examples: 
            # both = [
            #     {"role": "user", "content": ex["question"]},
            #     {"role": "assistant", "content": ex["answer"]}
            # ]
            question = [
                {"role": "user", "content": ex["question"]},
            ]

            answer = [
                {"role": "assistant", "content": ex["answer"]}
            ]
            # both_c = self.tokenizer.apply_chat_template(both, tokenize=False, add_generation_prompt=False)
            if "llama" in getattr(self.tokenizer, "name_or_path", "").lower(): 
                question_c = self.tokenizer.apply_chat_template(question, tokenize=False, add_generation_prompt=False) 
                answer_c = self.tokenizer.apply_chat_template(answer, tokenize=False, add_generation_prompt=False) 
                answer_c = "<|eot_id|>".join(answer_c.split("<|eot_id|>")[1:]) # get rid of system prompt
            else: 
                raise NotImplementedError()

            q_ids = self.tokenizer(question_c, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
            a_ids = self.tokenizer(answer_c, return_tensors="pt", add_special_tokens=False)["input_ids"][0]

            input_ids = torch.cat((q_ids , a_ids), dim=0)
            labels = torch.cat((torch.ones_like(q_ids, dtype=int) * -100, a_ids), dim=0)

            batch_input_ids.append(input_ids)
            batch_labels.append(labels) 
            batch_attn_mask.append(torch.ones_like(input_ids, dtype=int)) 

        all_input_ids = pad_sequence(batch_input_ids, batch_first=True, padding_value=self.tokenizer.eos_token_id, padding_side="left")
        all_labels = pad_sequence(batch_labels, batch_first=True, padding_value=-100, padding_side="left") 
        all_attention_masks = pad_sequence(batch_attn_mask, batch_first=True, padding_value=0, padding_side="left")

        return BatchEncoding({
            "input_ids": all_input_ids, 
            "labels": all_labels, 
            "attention_mask": all_attention_masks
        })

class LMCollator: 
    def __init__(self, tokenizer: PreTrainedTokenizer): 
        self.tokenizer = tokenizer
    def __call__(self, examples: Tuple) -> BatchEncoding: 

        batch_input_ids = []
        batch_labels = []
        batch_attn_mask = []
        for ex in examples: 
            # both = [
            #     {"role": "user", "content": ex["question"]},
            #     {"role": "assistant", "content": ex["answer"]}
            # ]
            # question = [
            #     {"role": "user", "content": ex["question"]},
            # ]

            # answer = [
            #     {"role": "assistant", "content": ex["answer"]}
            # ]
            text = ex["text"]
            # both_c = self.tokenizer.apply_chat_template(both, tokenize=False, add_generation_prompt=False)
            input_ids = self.tokenizer(text, return_tensors="pt", max_length=2048)["input_ids"][0]
            labels = deepcopy(input_ids)

            batch_input_ids.append(input_ids)
            batch_labels.append(labels) 
            batch_attn_mask.append(torch.ones_like(input_ids, dtype=int)) 

        all_input_ids = pad_sequence(batch_input_ids, batch_first=True, padding_value=self.tokenizer.eos_token_id, padding_side="left")
        all_labels = pad_sequence(batch_labels, batch_first=True, padding_value=-100, padding_side="left") 
        all_attention_masks = pad_sequence(batch_attn_mask, batch_first=True, padding_value=0, padding_side="left")

        return BatchEncoding({
            "input_ids": all_input_ids, 
            "labels": all_labels, 
            "attention_mask": all_attention_masks
        })


if __name__ == "__main__": 
    tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
    collator = LMCollator(tok)

    from torch.utils.data import DataLoader
    # from gsm import GSM8K 
    from pile import Pile 

    ds = Pile(["train", "test"])
    dl = DataLoader(ds["train"], batch_size=8, collate_fn=collator)
    next(iter(dl))