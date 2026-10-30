from .abstract import AbstractDataset
from .gsm import GSM8K 
from .pile import Pile

def TASK_MAP(name: str) -> AbstractDataset: 
    if "gsm" in name.lower(): 
        return GSM8K
    if "pile" in name.lower(): 
        return Pile