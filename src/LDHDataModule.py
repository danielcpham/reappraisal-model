
import numpy as np
import pytorch_lightning as lit
from torch.utils.data import SubsetRandomSampler, DataLoader
from transformers import AutoTokenizer


from .LDHData import LDHData

class LDHDataModule(lit.LightningDataModule):
    def __init__(self, batch_size):
        super().__init__()
        self.batch_size = batch_size
        self.tokenizer = AutoTokenizer.from_pretrained('distilbert-base-cased', use_fast=True)
        data = LDHData()
        data.load_training_data()
        data.load_eval_data()

        encoded_ds = data.train_dataset['obj'].map(
            lambda ds: self.tokenizer(ds['response'], add_special_tokens=True, padding="max_length", max_length=150)
        )

        encoded_ds.set_format(type='torch', columns=['input_ids', 'attention_mask', 'score'])

        dataset_size = len(encoded_ds)
        indices = list(range(dataset_size))
        split = int(np.floor(0.15 * dataset_size))
        np.random.shuffle(indices)
        train_indices, val_indices = indices[split:], indices[:split]

        # Creating PT data samplers and loaders:
        self.train_sampler = SubsetRandomSampler(train_indices)
        self.val_sampler = SubsetRandomSampler(val_indices)

        self.train_data = encoded_ds

    def setup(self, stage=None):
        # Try loading the dataset from disk. If we can't, reparse
        pass
            
    def train_dataloader(self):
        # TODO: remove obj setting
        return DataLoader(self.train_data, batch_size=self.batch_size, sampler=self.train_sampler)

    def val_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size, sampler=self.val_sampler)

    def test_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.batch_size)
