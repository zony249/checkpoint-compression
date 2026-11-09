import os  
from copy import deepcopy
from typing import List, Dict, Tuple

import torch 
from torch import nn 
from datasets import Dataset, load_dataset
from datasets.iterable_dataset import IterableDataset
from tqdm import tqdm
import pyarrow.dataset as pds
from pyarrow import CacheOptions


from .abstract import AbstractDataset 


class Pile(AbstractDataset): 
    def __init__(self, list_splits: List = [], **ds_kwargs): 
        super().__init__(list_splits, **ds_kwargs)

    def load_dataset(self) -> Dataset: 
        ds = load_dataset(
            "monology/pile-uncopyrighted",
            streaming=True, 
        )
        # ds = ds.shuffle(buffer_size=100000, seed=42).take(10000)
        return ds 

    def load_split(
        self,
        split:str,
        ds_temp: IterableDataset | None = None
    ) -> Dataset: 
        if ds_temp is not None:            
            if split == "train": 
                return ds_temp["train"].skip(1000)
            else: 
                stream_ds = ds_temp["train"].take(1000)
                return Dataset.from_generator(lambda: stream_ds)
        else:  
            ds_temp = self.dataset
            return ds_temp[split]
        

    def preprocess(self, split, dataset) -> Tuple[List[str], List[Dataset]]:
        # def extract_text(example): 
        #     return example["text"]
        # dataset = dataset.map(extract_text, batched=False)
        return (split, ), (dataset, ) 

    


if __name__ == "__main__": 

    pile = Pile(list_splits=["train"])
    pass