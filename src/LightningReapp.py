import os
import pdb
from pathlib import Path
from typing import Dict, List
import datasets
import numpy as np

import pandas as pd
import pytorch_lightning as lit
import torch
from nltk.tokenize import sent_tokenize
from sklearn.model_selection import train_test_split
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import random_split, DataLoader, SubsetRandomSampler
from transformers import AutoModel, AutoTokenizer
from transformers.models.distilbert.tokenization_distilbert import DistilBertTokenizer

from .LDHData import LDHData

class LightningReapp(lit.LightningModule):
  def __init__(self):
    super().__init__()
    self.model = AutoModel.from_pretrained('distilbert-base-cased')
    self.classifier = nn.Sequential(
      nn.Linear(768, 50),
      nn.ReLU(),
      #nn.Dropout(0.5),
      nn.Linear(50, 10),
      nn.ReLU()
    )

  def forward(self, input_ids, attention_mask):
    output = self.model(input_ids, attention_mask)
    last_hidden_state = output.last_hidden_state
    return self.classifier(last_hidden_state)

  def configure_optimizers(self):
    optimizer = optim.Adam(self.parameters(), lr=1e-3)
    return optimizer

  def training_step(self, batch, batch_idx):
    input_ids = batch['input_ids']
    attention_mask = batch['attention_mask']
    score = batch['score']
    output = self(input_ids, attention_mask)
    loss = F.mse_loss(output.sum(), score)
    return loss


class LDHDataModule(lit.LightningDataModule):
    def __init__(self, batch_size):
        super().__init__()
        self.batch_size = batch_size
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-cased', use_fast=True)
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
        print(self.train_data)
        return DataLoader(self.train_data, batch_size=self.batch_size, sampler=self.train_sampler)

    def val_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size, sampler=self.val_sampler)

    def test_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.batch_size)
