from typing import Tuple, List, Dict
from copy import deepcopy

import torch
from torch.utils.data import Dataset, Subset, random_split
from datasets import load_dataset


from .abstract import AbstractDataset 




class GSM8K(AbstractDataset): 
    def __init__(self, 
                 list_splits: str = ["train", "test"], 
                 hold_out_validation: bool = False): 
        self.hold_out_validation = hold_out_validation
        super().__init__(list_splits=list_splits)
    def load_dataset(self) -> Dataset: 
        ds_temp = load_dataset("openai/gsm8k", "main")
        return ds_temp
    def load_split(
        self,
        split:str,
        ds_temp: Dataset | None = None
    ):
        if ds_temp is None: 
            ds_temp = self.dataset
        return ds_temp[split]
    def preprocess(self, split:str, dataset: Dataset) -> Tuple[List[str], List[Dataset | Subset]]:
        if split == "train": 
            if self.hold_out_validation: 
                train_size = int(0.9 * dataset.num_rows) 
                val_size = dataset.num_rows - train_size 

                generator = torch.Generator().manual_seed(42)
                train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=generator)

                # t_mask = torch.ones(dataset.num_rows, dtype=bool) 
                # t_mask[val_thresh:] = False
                # v_mask = torch.ones_like(t_mask) 
                # v_mask[:val_thresh] = False

                # train_dataset = deepcopy(dataset)
                # val_dataset = deepcopy(dataset)
                return ("train", "validation"), (train_dataset, val_dataset)
            return ("train", ), (dataset,)
        else: 
            return (split,), (dataset,)
        return None, None

if __name__ == "__main__": 

    gsm = GSM8K(hold_out_validation=True)
    ds = load_dataset("openai/gsm8k", "main")
    train = ds["train"]

    pass