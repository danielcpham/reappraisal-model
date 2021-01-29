import os
import pickle as pkl
from collections import defaultdict
from typing import Dict, List, Tuple, Union


import numpy as np
import pandas as pd
from datasets import Dataset, DatasetDict
from nltk.tokenize import sent_tokenize
from pandas import DataFrame


class LDHData:
    def __init__(self, tokenizer=None):
        # TODO: convert to multi index
        self.dataframes = {
            'train': defaultdict(pd.DataFrame),
            'eval': defaultdict(pd.DataFrame)}
        # TODO: if exists pickles, grab them
        self.load_training_data()
        self.load_eval_data()
        # otherwise, load training_data and pickle 
        ## Save to disk if we don't already have a version
    # Functions to load data; parses if not available

    @property
    def datasets(self, *args):
        return to_datasets(self.dataframes)
        
    def load_training_data(self) -> None:
        if os.path.exists("./src/training/far.pkl") and os.path.exists("./src/training/obj.pkl"):
            self.dataframes['train'] = {
                'far': pd.read_pickle("./src/training/far.pkl"),
                'obj': pd.read_pickle("./src/training/obj.pkl")
            } 
            self.pickle_data()
        self._parse_training_data()

    def load_eval_data(self) -> None:
        if os.path.exists("./src/eval/far.pkl") and os.path.exists("./src/eval/obj.pkl"):
            self.dataframes['eval'] = {
                'far': pd.read_pickle("./src/eval/far.pkl"),
                'obj': pd.read_pickle("./src/eval/obj.pkl")
            } 
            self.pickle_data()
        self._parse_eval_data()

    def collapse_eval_data(self, df: DataFrame):
        """Let df be the dataframe obtained from loading evaluation data. 
        Expand the text in 'response' to have a single sentence per response.
        """
        new_responses = df['response'].map(lambda resp: _expand_response(resp))\
        .apply(pd.Series).unstack().reset_index()\
        .drop('level_1', axis=1)        
        collapsed = df.merge(new_responses, right_on='level_0', left_on=df.index, how="right")
        collapsed = collapsed.drop(['Condition', 'response', '__index_level_0__'], axis=1).rename(columns={0: 'response'}).dropna(subset=["addcode", 'response'])
        collapsed = collapsed[collapsed['response'] != "."]
        return collapsed

    def encode_training_data(self, data: DataFrame):
        dataset = Dataset.from_pandas(data)
        encoded_ds = dataset.map(
            lambda batch: self.tokenizer(
                batch['response'],
                add_special_tokens=True,
                padding="max_length",
                truncation=True))
        encoded_ds.set_format(type='torch', output_all_columns=True)
        return encoded_ds

    def encode_eval_data(self, data: DataFrame):
        dataset = Dataset.from_pandas(data)
        encoded_ds = dataset.map(
            lambda batch: self.tokenizer(
                batch['response'],
                add_special_tokens=True,
                padding="max_length",
                truncation=True))
        encoded_ds.set_format(type='torch', output_all_columns=True)
        return encoded_ds

    # Functions to read the data files directly
    def _parse_training_data(self) -> None:
        datadir_train = os.path.join(os.getcwd(),"./src/training")
        files = os.listdir(datadir_train)
        dfs: List[DataFrame] = []
        for file in files:
            if file.endswith(".csv"):
                filename = os.path.join(datadir_train, file)
                dfs.append(pd.read_csv(
                    filename, 
                    header = 0,
                    names=['response', "spatiotemp", "obj"]
                ))
        ldh : DataFrame = pd.concat(dfs, ignore_index=True)
        train_far_data = ldh[['response', 'spatiotemp']].rename(columns={'spatiotemp': 'score'})
        train_obj_data = ldh[['response', 'obj']].rename(columns={'obj': 'score'})
        self.dataframes['train'] = {
            'far': train_far_data,
            'obj': train_obj_data
            }

    def _parse_eval_data(self) -> None:
        datadir_eval = os.path.join(os.getcwd(), "./src/eval")
        ## TODO: If we already have a lock file, just return that
        columns = ["addcode", "Condition", "TextResponse"]
        # Read the excel files
        eval_far_data = pd.read_excel(os.path.join(datadir_eval, "Alg_Far_NEW.xlsx" ), usecols=columns, engine="openpyxl")\
            .rename(columns={"TextResponse": "response"})
        eval_obj_data = pd.read_excel(os.path.join(datadir_eval, "Alg_Obj_NEW.xlsx" ), usecols=columns, engine="openpyxl")\
            .rename(columns={"TextResponse": "response"})
        eval_far_data = eval_far_data[eval_far_data['response'].notna()]
        eval_obj_data = eval_obj_data[eval_obj_data['response'].notna()]
        self.dataframes['eval'] = {
            "far": self.collapse_eval_data(eval_far_data), 
            "obj": self.collapse_eval_data(eval_obj_data)
        }

    # Functions to read and save compressed data files
    def pickle_data(self) -> None:
        self.dataframes['train']['far'].to_pickle("./src/training/far.pkl")
        self.dataframes['train']['obj'].to_pickle("./src/training/obj.pkl")
        self.dataframes['eval']['far'].to_pickle("./src/eval/far.pkl")
        self.dataframes['eval']['obj'].to_pickle("./src/eval/obj.pkl")


    
    
def _expand_response(input_response: str) -> List[str]:
    sentences = sent_tokenize(input_response)
    return sentences

def to_datasets(dfs):
    return {
        'train': {
            'far': Dataset.from_pandas(dfs['train']['far']),
            'obj': Dataset.from_pandas(dfs['train']['obj'])
        }, 
        'eval': {
            'far': Dataset.from_pandas(dfs['eval']['far']),
            'obj': Dataset.from_pandas(dfs['eval']['obj'])
        }
    }
