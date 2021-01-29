import os
from typing import Dict

import numpy as np
import pandas as pd
from datasets import Dataset, Features, Sequence, Value
from nltk.tokenize import sent_tokenize

TRAIN_FEATURES = Features({
                'attention_mask': Sequence(feature=Value(dtype='int64', id=None), length=-1, id=None),
                'input_ids': Sequence(feature=Value(dtype='int64', id=None), length=-1, id=None),
                'response': Value(dtype='string', id=None),
                'score': Value(dtype='float32', id=None)
            })

class LDHData:
    def __init__(self, tokenizer=None):
        if tokenizer:
            self.tokenizer = tokenizer
            self.tokenizer.model_input_names.append("score")
        # Read training data into a single dataframe
        datadir_train = os.path.join(os.getcwd(),"./src/training")
        files = os.listdir(datadir_train)
        dfs = []
        for file in files:
            if file.endswith(".csv"):
                filename = os.path.join(datadir_train, file)
                dfs.append(pd.read_csv(
                    filename, 
                    header = 0,
                    names=['response', "spatiotemp", "obj"]
                ))
        ldh = pd.concat(dfs, ignore_index=True)

        train_far_data =  Dataset.from_pandas(ldh[['response', 'spatiotemp']])
        ldh_train_obj = Dataset.from_pandas(ldh[['response', 'obj']])
        train_far_data.rename_column_('spatiotemp', 'score')
        ldh_train_obj.rename_column_('obj', 'score')
        
        train_far_data = train_far_data
        train_obj_data = ldh_train_obj

        # Read evaluation data and save as dataframes
        datadir_eval = os.path.join(os.getcwd(), "./src/eval")
        columns = ["addcode", "Subj_ID", "Condition", "TextResponse"]

        eval_far_data = pd.read_excel(os.path.join(datadir_eval, "Alg_Far_NEW.xlsx" ), usecols=columns, engine="openpyxl")
        eval_obj_data = pd.read_excel(os.path.join(datadir_eval, "Alg_Obj_NEW.xlsx" ), usecols=columns, engine="openpyxl")

        eval_far_data.rename(columns={"TextResponse": "response"}, inplace=True)
        eval_obj_data.rename(columns={"TextResponse": "response"}, inplace=True)

        eval_far_data = eval_far_data[eval_far_data['response'].notna()]
        eval_obj_data = eval_obj_data[eval_obj_data['response'].notna()]

        self.data = {
            'train': {
                'far': train_far_data,
                'obj': train_obj_data
            }, 
            'eval': {
                'far': eval_far_data,
                'obj': eval_obj_data
            }
        }

    def set_tokenizer(self, tokenizer):
        self.tokenizer = tokenizer

    def get_train_far_data(self, encoded=False):
        if encoded and self.tokenizer:
            return self.encode_training_data(self.data['train']['far'])
        return self.data['train']['far']
    
    def get_train_obj_data(self, encoded=False):
        if encoded and self.tokenizer:
            return self.encode_training_data(self.data['train']['obj'])
        return self.data['train']['obj']

    def get_eval_far_data(self, encoded=False):
        if encoded and self.tokenizer:
            return self.encode_eval_data(self.data['eval']['far'])
        return self.data['eval']['far']

    def get_eval_obj_data(self, encoded=False):
        if encoded and self.tokenizer:
            return self.encode_eval_data(self.data['eval']['obj'])
        return self.data['eval']['obj']
    
    def encode_training_data(self, dataset: Dataset):
        print(dataset)
        encoded_ds = dataset.map(
            lambda batch: self.tokenizer(
                batch['response'],
                add_special_tokens=True,
                padding="max_length",
                truncation=True), 
            batched=True, batch_size=16)
        encoded_ds.set_format(type='torch', output_all_columns=True)
        return encoded_ds

    def encode_eval_data(self, dataset: Dataset):
        print(dataset)
        encoded_ds = dataset.map(
            lambda batch: self.tokenizer(
                batch['response'],
                add_special_tokens=True,
                padding="max_length",
                truncation=True), 
            batched=True, batch_size=16)
        encoded_ds.set_format(type='torch', output_all_columns=True)

        return encoded_ds