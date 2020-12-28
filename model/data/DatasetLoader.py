import os
from re import match
from typing import Text, Union
import pandas as pd

from datasets import load_dataset, DatasetDict, Dataset

# https://github.com/huggingface/datasets/blob/master/templates/new_dataset_script.py

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

