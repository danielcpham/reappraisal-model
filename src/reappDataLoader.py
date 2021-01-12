from typing import Union

import pandas as pd
from datasets import Dataset, DatasetDict, load_dataset
from transformers import BatchEncoding, PreTrainedTokenizer
# Utility functions for creating dataset loaders.
# DatasetLoaders are iterators that return each point in the dataset.
# https://github.com/huggingface/datasets/blob/master/templates/new_dataset_script.py

def encode_data(tokenizer: PreTrainedTokenizer, dataset:Dataset, text_col_name: str) -> BatchEncoding:
  # check that the dataset is not already encoded
  # if already encoded, just return it
  encoded_dataset = dataset.map(
    lambda batch: tokenizer(batch[text_col_name],
      add_special_tokens = True,
      padding=True,
      truncation = True), batched=True)
  encoded_dataset.set_format(type='torch', output_all_columns=True)
  return encoded_dataset



Data = Union[Dataset, DatasetDict]
class InvalidModelNameException(Exception):
  """When an model name is passed for a model that doesn't exist."""
  pass 

"""
Single entry point to pick up datasets.
"""
def get_dataset(dataset_name: str) -> Data:
  # TODO: match dataset on name

  if dataset_name == 'emobank':
    return load_emobank()
  elif dataset_name == 'ldh':
    return load_dataset('./input/LDHDataset.py')
  else:
    raise InvalidModelNameException

def load_emobank() -> Data:
  return load_dataset("csv", data_files=['./input/emobank.csv'])


