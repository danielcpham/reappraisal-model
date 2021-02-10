import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import pandas as pd
from datasets import Dataset, DatasetDict
from nltk.tokenize import sent_tokenize


class LDHData:
    def __init__(
        self,
        tokenizer,
        input_dir: os.PathLike = Path.cwd(),
        output_dir: os.PathLike = Path.cwd(),
    ):
        """Initializes the directories from which the data will be coming and leaving.

        Args:
            input_dir (os.PathLike, optional): Where the input data is contained (training data). Defaults to None.
            output_dir (os.PathLike, optional): Where the cached datasets are saved. Defaults to None.
        """
        self.tokenizer = tokenizer
        self.train_dir = (
            Path(input_dir) / "data" / "training"
        )  # Where the training data is coming from
        self.eval_dir = Path(input_dir) / "data" / "eval"  # Test data
        # TODO: assert that the directories are valid.
        self.datasets = defaultdict(str)
        self.save_dir = Path(output_dir)  # Where cached data is saved.

    def encode(self, ds, ds_output_dir):
        encoded_ds = self.tokenizer(
            ds["response"],
            add_special_tokens=True,
            padding="max_length",
            max_length=150,
        )
        encoded_ds.set_format(
            type="torch",
            columns=["input_ids", "attention_mask", "score"],
            output_all_columns=True,
        )
        encoded_ds.save_to_disk(ds_output_dir)
        return encoded_ds

    @property
    def train_dataset(self):
        return self.datasets["train"]

    @property
    def eval_dataset(self):
        return self.datasets["eval"]

    def load_training_data(
        self, encoded=True, s_pandas=False, save_datasets=True
    ) -> None:
        training_save_dir = self.save_dir / "training"
        try:
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
            if encoded:
                self.datasets["train"] = self.datasets["train"].map(
                    lambda ds: self.encode(ds, training_save_dir)
                )
            if save_datasets:
                print(f"Saving training dataset to {training_save_dir}")
                self.datasets["train"].save_to_disk(training_save_dir)

    def load_eval_data(self, encoded=True, as_pandas=False, save_datasets=True) -> None:
        eval_save_dir = self.save_dir / "training"
        try:
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
            if encoded:
                self.datasets["eval"] = self.datasets["eval"].map(
                    lambda ds: self.encode(ds, eval_save_dir)
                )
            if save_datasets:
                print(f"Saving evaluation dataset to {eval_save_dir}")
                self.datasets["eval"].save_to_disk(eval_save_dir)

    def collapse_eval_data(self, df: pd.DataFrame):
        """Let df be the dataframe obtained from loading evaluation data.
        Expand the text in 'response' to have a single sentence per response.
        """
        new_responses = (
            df["response"]
            .map(lambda resp: _expand_response(resp))
            .apply(pd.Series)
            .unstack()
            .reset_index()
            .drop("level_1", axis=1)
        )
        collapsed = df.merge(
            new_responses, right_on="level_0", left_on=df.index, how="right"
        )
        collapsed = (
            collapsed.drop(["Condition", "response"], axis=1)
            .rename(columns={0: "response"})
            .dropna(subset=["addcode", "response"])
        )
        collapsed = collapsed[collapsed["response"] != "."]
        return collapsed

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
        columns = ["addcode", "Condition", "TextResponse"]
        # Read the excel files
        eval_far_data = pd.read_excel(
            os.path.join(self.eval_dir, "Alg_Far_NEW.xlsx"),
            usecols=columns,
            engine="openpyxl",
        ).rename(columns={"TextResponse": "response"})
        eval_obj_data = pd.read_excel(
            os.path.join(self.eval_dir, "Alg_Obj_NEW.xlsx"),
            usecols=columns,
            engine="openpyxl",
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
