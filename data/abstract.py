from typing import List, Dict, Tuple
from abc import ABC 

import torch 
import torch.nn
from torch.utils.data import Dataset 

class AbstractDataset(ABC): 
    def __init__(self, list_splits: List): 

        ds_temp = self.load_dataset() 
        self.dataset = {}
        for item in list_splits: 
            self.dataset[item] = self.load_split(item, ds_temp)

        for item in list_splits: 
            splits, datasets = self.preprocess(item, self.dataset[item]) 
            for s, d in zip(splits, datasets): 
                self.dataset[s] = d   
        del ds_temp

    def load_dataset(self) -> Dataset: 
        raise NotImplementedError()
    def load_split(self, split:str) -> Dataset: 
        raise NotImplementedError() 
    def preprocess(self, split:str, dataset: Dataset) -> Dataset:
        raise NotImplementedError()
    def get_splits(self): 
        return list(self.dataset.keys())
    def __getitem__(self, idx): 
        return self.dataset[idx]