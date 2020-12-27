import pandas as pd

from typing import Dict, List
# TODO: When testing, make sure we can run on different models, although DistilBERT should work fine.
from transformers import BatchEncoding, Trainer, TrainingArguments
from datasets import Dataset

from .ModelLoader import load
from input.data.DatasetLoader import get_dataset


# 1. Load Training Data
ldh = get_dataset('ldh')
emobank = get_dataset('emobank')

# 2. Load PreTrained models
from model.ModelLoader import load
model, tokenizer = load('distilbert-base-uncased')

def tokenize_batch(batch: BatchEncoding, text_col_name: str):
  return tokenizer(batch[text_col_name],
                  add_special_tokens = True,
                  padding = True,
                  truncation = True)

def encode(dataset: Dataset, batch_size: int, columns:List[str]):
  return None

# 3. Finetune Model
# TODO: generate Training Arguments
# TODO: generate Trainer
# TODO: should we do this in native PyTorch instead?




"""
Wrapper class for pretrained transformer models.
"""
# class TransformerModel:
#   def __init__(self, model_name, datasets:List[Dataset]):
#     try:
#       self.name = model_name
#       self.model = DistilBertModel.from_pretrained(model_name)
#       self.tokenizer = DistilBertTokenizerFast.from_pretrained(model_name)
#     except:
#       # TODO: Real error exception handling here
#       print("ERROR: Not a proper model")
#     self.training_args: TrainingArguments

#   def tokenize(self, batch):
#     return self.tokenizer(batch['text'],
#       add_special_tokens=True,            # Add CLS and SEP tokens. 
#       padding=True,                       # Pad to the longest sequence in the batch.
#       truncation=True)

#   def encode(self, dataset: pd.DataFrame, batch_size: int, columns:List[str]) -> BatchEncoding :
#     encoded_dataset = dataset.map(self.tokenize(batch_size), batch_size=batch_size)
#     # TODO: check to see if there's somewhere else I can do this (probably the tokenizer??)
#     # TODO: Make sure the column labels are correct for the task we're doing
#     encoded_dataset.set_format('torch', columns=columns) 
#     return encoded_dataset
  
#   def set_training_args(self) -> None:
#     # TODO: for each key in the dict
#     # check that the key is a valid argument for trainingArguments
#     # if not, log/ignore.
#     self.training_args = TrainingArguments(
#       output_dir='./results',          
#       num_train_epochs=1,              
#       per_device_train_batch_size=16,
#       per_device_eval_batch_size=64,
#       warmup_steps=500,                # number of warmup steps for learning rate scheduler
#       weight_decay=0.01,               # strength of weight decay
#       logging_dir='./logs'             # directory for storing logs
#     )
  
#   """
#   Fine tuning step: Given a training set and a validation dataset, train the model
#   """
#   def train(self, train_dataset:BatchEncoding, eval_dataset:BatchEncoding):
#     trainer = Trainer(
#       model = self.model, # TODO: null check here
#       args = self.training_args,
#       train_dataset = train_dataset,
#       eval_dataset = eval_dataset
#     )
#     # TODO: Loss function and optimizer?

# def load_pretrained(self, model_name):
#   model = DistilBertModel.from_pretrained(model_name)
#   tokenizer = DistilBertTokenizerFast.from_pretrained(model_name)
#   return model, tokenizer

# def finetune(model: DistilBertModel):
#   return None

# def train(trainer:Trainer):
#   return None

# def eval():
#   return None

# # Features to extract

# # We can use VAD to generate emotion scores for sentiment analysis
# # - Valence
# # - Arousal
# # - Dominance 
# # We can use 
