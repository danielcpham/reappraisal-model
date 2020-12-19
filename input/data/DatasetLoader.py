from typing import Tuple, Union 
import numpy as np
from numpy.lib.npyio import load
import pandas as pd
from datasets import ReadInstruction, Dataset
from transformers import PreTrainedModel, PreTrainedTokenizer, Trainer, training_args
from sklearn.model_selection import train_test_split 
#TODO: Use the huggingface one instead

# https://github.com/huggingface/datasets/blob/master/templates/new_dataset_script.py

"""
Class grouping LDH related tasks together.
- Loading the data into the model
- Converting the data into a dataset
- Exposing splitting for k fold cv
"""
class LDHDatasetLoader:
  """
  Wrapper for load_dataset for loading LDH model.
  """
  def __init__(self, test_size:float, split_string):
    #TODO: this could be done directly in load_dataset but let's use the DF for debugging.
    self.df = pd.concat([
      pd.read_csv("./eval/data_test_example.csv"),
      pd.read_csv("./eval/data_train_example.csv")
    ])
    self.df.rename(columns={'Unnamed: 0': 'serial'}, inplace=True)
    self.train_df = self.df

    # https://towardsdatascience.com/fine-tuning-a-bert-model-with-transformers-c8e49c4e008b
    train_responses, val_responses, train_faraway, val_faraway, train_obj, val_obj = train_test_split(
      self.df['response'].tolist(),
      self.df['spatiotemp'].tolist(),
      self.df['obj'].tolist()
    )

    #TODO: Create a custom dataset.



# Encode training data 
# sentiment analysis is evaluated on precision, recall, f1, and accuracy

class EmoBankDatasetLoader:
  def __init__(self):
    self.df = pd.read_csv('./input/emobank.csv')
    self.train_df = self.df[self.df['split'] == 'train']
    self.test_df = self.df[self.df['split'] == 'test']

    # Dataset Objects
    self.train_dataset = Dataset.from_pandas(self.df)
    #TODO: on process, save the encoded dataset and reload with `load_from_disk`
  

  

