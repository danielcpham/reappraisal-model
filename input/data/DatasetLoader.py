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
  datasets = {
    'emobank' : load_emobank(),
    'ldh' : load_dataset('input/data/LDHDataset.py')
  }

  if dataset_name not in datasets:
    raise InvalidModelNameException

  return datasets[dataset_name]

def load_emobank() -> Data:
  return load_dataset("csv", datafiles='./input/emobank.csv')

