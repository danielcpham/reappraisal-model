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
        self.train_dir = config.train_dir
        self.eval_dir = config.eval_dir
        self.datasets = defaultdict(str)

    
    def load_training_data(self, as_pandas=False, save_datasets=True) -> None:
        # Try loading the dataset from disk. If we can't, reparse
        try:
            self.datasets['train'] = DatasetDict.load_from_disk(self.train_dir)
            print("Training data loaded from disk.")
        except:
            print("Regenerating training data.")
            train_df_dict = self._parse_training_data(self.train_dir)
            self.datasets['train'] = DatasetDict({
                'far': Dataset.from_pandas(train_df_dict['far']),
                'obj' : Dataset.from_pandas(train_df_dict['obj'])
            })
            if save_datasets:
                print(f"Saving training dataset to {self.train_dir}")
                self.datasets['train'].save_to_disk(self.train_dir)

    def load_eval_data(self, as_pandas=False, save_datasets=True) -> None:
        try:
            self.datasets['eval'] = DatasetDict.load_from_disk(self.eval_dir)
            print("Evaluation data loaded from disk.")
        except:
            print("Regenerating evaluation data.")
            eval_df_dict = self._parse_eval_data(self.eval_dir)
            self.datasets['eval'] = DatasetDict({
                'far': Dataset.from_pandas(eval_df_dict['far']),
                'obj' : Dataset.from_pandas(eval_df_dict['obj'])
            })
            if save_datasets:
                print(f"Saving evaluation dataset to {self.eval_dir}")
                self.datasets['eval'].save_to_disk(self.eval_dir)

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
        ds.map(lambda batch: tokenizer(
                batch['response'],
                add_special_tokens=True,
                padding="max_length",
                truncation=True), **tokenizer_args)
        ds.set_format(type="torch", output_all_columns=True)
        return ds

    # Functions to read the data files directly
    def _parse_training_data(self, train_dir: str) -> Dict[str, pd.DataFrame]:
        files = os.listdir(train_dir)
        dfs: List[DataFrame] = []
        for file in files:
            if file.endswith(".csv"):
                filename = os.path.join(train_dir, file)
                dfs.append(pd.read_csv(
                    filename, 
                    header = 0,
                    names=['response', "spatiotemp", "obj"]
                ))
        ldh : DataFrame = pd.concat(dfs, ignore_index=True)
        train_far_data = ldh[['response', 'spatiotemp']].rename(columns={'spatiotemp': 'score'})
        train_obj_data = ldh[['response', 'obj']].rename(columns={'obj': 'score'})
        return {
            'far': train_far_data,
            'obj': train_obj_data
            }

    def _parse_eval_data(self, eval_dir: str) -> Dict[str, pd.DataFrame]:
        columns = ["addcode", "Condition", "TextResponse"]
        # Read the excel files
        eval_far_data = pd.read_excel(os.path.join(self.eval_dir, "Alg_Far_NEW.xlsx" ), usecols=columns, engine="openpyxl")\
            .rename(columns={"TextResponse": "response"})
        eval_obj_data = pd.read_excel(os.path.join(self.eval_dir, "Alg_Obj_NEW.xlsx" ), usecols=columns, engine="openpyxl")\
            .rename(columns={"TextResponse": "response"})
        eval_far_data = eval_far_data[eval_far_data['response'].notna()]
        eval_obj_data = eval_obj_data[eval_obj_data['response'].notna()]
        return {
            "far": self.collapse_eval_data(eval_far_data), 
            "obj": self.collapse_eval_data(eval_obj_data)
        }

def _expand_response(input_response: str) -> List[str]:
    sentences = sent_tokenize(input_response)
    return sentences

