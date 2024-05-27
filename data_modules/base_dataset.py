from abc import ABC, abstractmethod
import logging
import math
import os
import pickle
from typing import List, Tuple
import numpy as np
import torch
from torch.utils.data.dataset import Dataset
from tqdm import tqdm

from data_modules.input_example import InputExample


class BaseDataset(Dataset, ABC):
    """
    Base class for all datasets.
    """
    name = None         # name of the dataset


    def __init__(
        self,
        data_dir: str,
        fold: int = 0,
        split: str = 'train',
    ):
        self.data_path = data_dir
        loaded_dataset = self.load_data(self.data_path, load_fold=fold)
        print(f"Loading {split} of fold {fold} ....")
        self.examples = loaded_dataset[fold][split]
        self.features = self.compute_features(self.examples)
        self.size = len(self.features)
        print(f"{split} of fold {fold} has {self.size} datapoints!")
    
    def __len__(self) -> int:
        return self.size
    
    def __getitem__(self, index):
        return self.features[index]

    @abstractmethod
    def load_schema(self):
        """
        Load extra dataset information, such as entity/relation types.
        """
        pass

    @abstractmethod
    def load_data(self, split: str, data_path: str, load_fold: int) -> List[InputExample]:
        """
        Load data for a single split (train, dev, or test).
        """
        pass

    def compute_features(self, examples: List[InputExample]) :
        """
        Compute features for model 
        """
        return examples
    

    def my_collate(self, batch: List[InputExample]):
        return batch


        
