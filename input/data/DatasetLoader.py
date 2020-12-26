from re import match
from typing import Union
import pandas as pd

# https://github.com/huggingface/datasets/blob/master/templates/new_dataset_script.py

class InvalidModelNameException(Exception):
  """When an model name is passed for a model that doesn't exist."""
  pass 

"""
Single entry point to pick up datasets.
"""
def get_dataset(dataset_name: str) -> pd.DataFrame:
  # TODO: match dataset on name
  datasets = {
    'emobank' : load_emobank(),
    'ldh' : load_ldh()
  }

  if dataset_name not in datasets:
    raise InvalidModelNameException

  return datasets[dataset_name]

def load_ldh() -> pd.DataFrame:
  #TODO: return as a loaded Transformers Dataset object
  # TODO: Clean data
  df = pd.concat([
    pd.read_csv("./eval/data_test_example.csv"),
    pd.read_csv("./eval/data_train_example.csv")
  ])
  df.rename(columns={'Unnamed: 0': 'serial'}, inplace=True)
  return df

def load_emobank():
  return pd.read_csv('./input/emobank.csv')


  

