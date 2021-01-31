import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Union

import numpy as np
import pandas as pd
from datasets import Dataset, DatasetDict
from nltk.tokenize import sent_tokenize
from pandas import DataFrame
from transformers import PreTrainedTokenizer

from .LDHConfig import LDHConfig

class LDHData:
    def __init__(self, tokenizer=None, config: LDHConfig = None):
        if not config:
            config = LDHConfig.make()
        self.training_dir = config.training_dir
        self.eval_dir = config.eval_dir
        self.dataframes = {
            'train': defaultdict(pd.DataFrame),
            'eval': defaultdict(pd.DataFrame)}
        self.load_training_data()
        self.load_eval_data()

    @property
    def datasets(self, *args):
        return to_datasets(self.dataframes)
    
    def load_training_data(self) -> None:
        self._parse_training_data()

    def load_eval_data(self) -> None:
        self._parse_eval_data()

    def collapse_eval_data(self, df: DataFrame):
        """Let df be the dataframe obtained from loading evaluation data. 
        Expand the text in 'response' to have a single sentence per response.
        """
        new_responses = df['response'].map(lambda resp: _expand_response(resp))\
        .apply(pd.Series).unstack().reset_index()\
        .drop('level_1', axis=1)        
        collapsed = df.merge(new_responses, right_on='level_0', left_on=df.index, how="right")
        collapsed = collapsed.drop(['Condition', 'response'], axis=1).rename(columns={0: 'response'}).dropna(subset=["addcode", 'response'])
        collapsed = collapsed[collapsed['response'] != "."]
        return collapsed
    
    def encode_datasets(self, tokenizer, **tokenizer_args) -> Dict:
        ds = self.datasets
        for key, value in ds.items():
            ds[key] = value.map(lambda batch: tokenizer(
                batch['response'],
                add_special_tokens=True,
                padding="max_length",
                truncation=True), **tokenizer_args)
            ds[key].set_format(type="torch", output_all_columns=True)
        return ds

    # Functions to read the data files directly
    def _parse_training_data(self) -> None:
        files = os.listdir(self.training_dir)
        dfs: List[DataFrame] = []
        for file in files:
            if file.endswith(".csv"):
                filename = os.path.join(self.training_dir, file)
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
        columns = ["addcode", "Condition", "TextResponse"]
        # Read the excel files
        eval_far_data = pd.read_excel(os.path.join(self.eval_dir, "Alg_Far_NEW.xlsx" ), usecols=columns, engine="openpyxl")\
            .rename(columns={"TextResponse": "response"})
        eval_obj_data = pd.read_excel(os.path.join(self.eval_dir, "Alg_Obj_NEW.xlsx" ), usecols=columns, engine="openpyxl")\
            .rename(columns={"TextResponse": "response"})
        eval_far_data = eval_far_data[eval_far_data['response'].notna()]
        eval_obj_data = eval_obj_data[eval_obj_data['response'].notna()]
        self.dataframes['eval'] = {
            "far": self.collapse_eval_data(eval_far_data), 
            "obj": self.collapse_eval_data(eval_obj_data)
        }

    # # Functions to read and save compressed data files
    # def pickle_data(self) -> None:
    #     try: # downgrade to protocol 4 for python 3.6 compatibility
    #         self.dataframes['train']['far'].to_pickle(Path.join(self.training_dir, "far.pkl"), protocol=4)
    #         self.dataframes['train']['obj'].to_pickle(Path.join(self.training_dir, "obj.pkl"), protocol=4)
    #         self.dataframes['eval']['far'].to_pickle(Path.join(self.training_dir, "far.pkl"), protocol=4)
    #         self.dataframes['eval']['obj'].to_pickle(Path.join(self.training_dir, "obj.pkl"), protocol=4)
    #     except:
    #         pass 
def _expand_response(input_response: str) -> List[str]:
    sentences = sent_tokenize(input_response)
    return sentences

def to_datasets(dfs) -> Dict[str, DatasetDict]:
    return {
        'train': DatasetDict({
            'far': Dataset.from_pandas(dfs['train']['far']),
            'obj': Dataset.from_pandas(dfs['train']['obj'])
        }), 
        'eval': DatasetDict({
            'far': Dataset.from_pandas(dfs['eval']['far']),
            'obj': Dataset.from_pandas(dfs['eval']['obj'])
        })
    }

