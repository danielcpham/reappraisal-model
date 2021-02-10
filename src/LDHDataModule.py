from collections import namedtuple
from types import MethodType
from typing import Generator, List, Tuple

import numpy as np
import pytorch_lightning as lit
from datasets import Dataset, DatasetDict
from sklearn.model_selection import GroupKFold
from torch.utils.data import RandomSampler, DataLoader
from torch.utils.data.dataset import Subset
from transformers import AutoTokenizer

from .LDHData import LDHData

DEFAULT_TOKENIZER = AutoTokenizer.from_pretrained(
            "distilbert-base-uncased-finetuned-sst-2-english", use_fast=True)

class LDHDataModule(lit.LightningDataModule):
    def __init__(self, batch_size=16, tokenizer=DEFAULT_TOKENIZER, strat='obj', kfolds=5):
        super().__init__()
        self.strat = strat
        self.batch_size = batch_size
        self.tokenizer = tokenizer
        self.kfolds = kfolds
        self.current_split = 0

        data = LDHData(self.tokenizer) 
        data.load_training_data()
        train_data: Dataset = data.train_dataset[self.strat] 
        self.responses = train_data['response'] # * We probably don't need this???
        encoded_ds = train_data.map(
            lambda ds: self.tokenizer(
                ds["response"],
                add_special_tokens=True,
                padding="max_length",
                max_length=150,
            )
        )
        encoded_ds.set_format(
            type="torch", columns=["input_ids", "attention_mask", "score"]
        )

        self.train_data = encoded_ds

         # Create k-fold training_data indices for cross validation
        indices = range(len(self.train_data))
        indices = np.arange(len(self.train_data))
        indices = indices % self.kfolds
        np.random.shuffle(indices)
        cv = GroupKFold(self.kfolds)
        self.splits = list(cv.split(self.train_data, groups=indices))

    def get_train_dataloader(self, split: int):
        train_idx = self.splits[split][0].tolist() # retrieve the split generated by GroupKFold
        data = Subset(self.train_data, train_idx)
        return DataLoader(data, batch_size=self.batch_size)
    
    def get_val_dataloader(self, split: int):
        val_idx = self.splits[split][1].tolist() # retrieve the split generated by GroupKFold
        data = Subset(self.train_data, val_idx)
        return DataLoader(data, batch_size=self.batch_size)

    def train_dataloader(self):
        return self.get_train_dataloader(self.current_split)

    def val_dataloader(self):
        return self.get_val_dataloader(self.current_split)

