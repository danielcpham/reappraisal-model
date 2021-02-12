import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from datasets import Dataset, DatasetDict
from nltk.tokenize import sent_tokenize


class LDHData:
    def __init__(
        self,
        tokenizer,
        input_dir: os.PathLike = Path.cwd() / 'data',
        output_dir: os.PathLike = Path.cwd() / 'output',
    ):
        """Initializes the directories from which the data will be coming and leaving.

        Args:
            input_dir (os.PathLike, optional): Where the input data is contained (training data). Defaults to None.
            output_dir (os.PathLike, optional): Where the cached datasets are saved. Defaults to None.
        """
        self.tokenizer = tokenizer
        self.train_dir = (
            Path(input_dir) / "training"
        )  # Where the training data is coming from
        self.eval_dir = Path(input_dir) / "eval"  # Test data
        # TODO: assert that the directories are valid.
        self.datasets = defaultdict(str)
        self.save_dir = Path(output_dir)  # Where cached data is saved.

    @property
    def train_dataset(self):
        return self.datasets["train"]

    @property
    def eval_dataset(self):
        return self.datasets["eval"]

    def load_training_data(
        self, force_reload=False, save_datasets=True
    ) -> None:
        training_save_dir = self.save_dir / "training"
        try:
            if force_reload:
                raise Exception()
            # If the training data has already been save, load it from the save_directory
            self.datasets["train"] = DatasetDict.load_from_disk(training_save_dir)
            print("Training data loaded from disk.")
        except:
            # If it hasn't regenerate the training data.
            print("Regenerating training data.")
            train_df_dict = self._parse_training_data(self.train_dir)
            self.datasets["train"] = DatasetDict(
                {
                    "far": Dataset.from_pandas(train_df_dict["far"]),
                    "obj": Dataset.from_pandas(train_df_dict["obj"]),
                }
            )
            if save_datasets:
                print(f"Saving training dataset to {training_save_dir}")
                self.datasets["train"].save_to_disk(training_save_dir)

    def load_eval_data(self, force_reload=False, save_datasets=True) -> None:
        eval_save_dir = self.save_dir / "eval"
        try:
            if force_reload:
                raise Exception()
            self.datasets["eval"] = DatasetDict.load_from_disk(eval_save_dir)
            print("Evaluation data loaded from disk.")
        except:
            print("Regenerating evaluation data.")
            eval_df_dict = self._parse_eval_data(self.eval_dir)
            self.datasets["eval"] = DatasetDict(
                {
                    "far": Dataset.from_pandas(eval_df_dict["far"]),
                    "obj": Dataset.from_pandas(eval_df_dict["obj"]),
                }
            )
            if save_datasets:
                print(f"Saving evaluation dataset to {eval_save_dir}")
                self.datasets["eval"].save_to_disk(eval_save_dir)

    def collapse_eval_data(self, df: pd.DataFrame):
        """Let df be the dataframe obtained from loading evaluation data.
        Expand the text in 'response' to have a single sentence per response.
        """
        df['response'] = df['response'].map(sent_tokenize, na_action='ignore')
        texts = df['response'].dropna()
        lens_of_lists = texts.apply(len)
        origin_rows = range(texts.shape[0])
        destination_rows = np.repeat(origin_rows, lens_of_lists)
        non_list_cols = [idx for idx, col in enumerate(df.columns) 
                        if col != 'response']
        expanded_df = df.iloc[destination_rows, non_list_cols].copy()
        expanded_df['split_response'] = [i for items in texts
                                    for i in items]
        expanded_df = expanded_df[expanded_df['split_response'] != "."].reset_index(drop=True)
        assert expanded_df.apply(pd.unique)['daycode'].size == 5
        assert expanded_df.apply(pd.unique)['Condition'].size == 3
        expanded_df.rename(columns={'split_response': 'response'}, inplace=True)
        return expanded_df

    # Functions to read the data files directly
    def _parse_training_data(self, train_dir: str) -> Dict[str, pd.DataFrame]:
        files = os.listdir(train_dir)
        dfs: List[pd.DataFrame] = []
        for file in files:
            if file.endswith(".csv"):
                filename = os.path.join(train_dir, file)
                dfs.append(
                    pd.read_csv(
                        filename, header=0, names=["response", "spatiotemp", "obj"]
                    )
                )
        ldh: pd.DataFrame = pd.concat(dfs, ignore_index=True)
        train_far_data = ldh[["response", "spatiotemp"]].rename(
            columns={"spatiotemp": "score"}
        )
        train_obj_data = ldh[["response", "obj"]].rename(columns={"obj": "score"})
        return {"far": train_far_data, "obj": train_obj_data}

    def _parse_eval_data(self, eval_dir: str) -> Dict[str, pd.DataFrame]:
        # Read the excel files
        eval_far_data = pd.read_excel(
            os.path.join(self.eval_dir, "Alg_Far_NEW.xlsx"),
            engine="openpyxl"
        ).rename(columns={"TextResponse": "response"})
        eval_obj_data = pd.read_excel(
            os.path.join(self.eval_dir, "Alg_Obj_NEW.xlsx"),
            engine="openpyxl"
        ).rename(columns={"TextResponse": "response"})
        eval_far_data = eval_far_data[eval_far_data["response"].notna()]
        eval_obj_data = eval_obj_data[eval_obj_data["response"].notna()]
        return {
            "far": self.collapse_eval_data(eval_far_data),
            "obj": self.collapse_eval_data(eval_obj_data),
        }

def _expand_response(input_response: str) -> List[str]:
    sentences = sent_tokenize(input_response)
    return sentences
