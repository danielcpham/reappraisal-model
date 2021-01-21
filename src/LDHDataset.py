import os

import pandas as pd
import datasets
from datasets import load_dataset

# TODO: Add development config
class LDHConfig(datasets.BuilderConfig):
    """Builder Config for LDH Data"""

    def __init__(self, **kwargs):
        super(LDHConfig, self).__init__(**kwargs)


class LDHDataset(datasets.GeneratorBasedBuilder):
    # TODO: create configs for testing and training data
    BUILDER_CONFIGS = [
        LDHConfig(
            name="ldh"
        )
    ]

    def _info(self) -> datasets.DatasetInfo:
        return datasets.DatasetInfo(
            features=datasets.Features(
                {
                    "text": datasets.Value("string"),
                    "far": datasets.Value("float32"),
                    "obj": datasets.Value("float32")
                }
            )
        )

    def _split_generators(self, dl_manager):
        # TODO: move data online
        """
        The `data_files` kwarg in load_dataset() can be a str, List[str], Dict[str,str], or Dict[str,List[str]].
        If str or List[str], then the dataset returns only the 'train' split.
        If dict, then keys should be from the `datasets.Split` enum.

        https://github.com/huggingface/datasets/blob/master/datasets/text/text.py
        """
        if not self.config.data_files:
            raise ValueError(f"At least one data file must be specified, but got data_files={self.config.data_files}")
        data_files = dl_manager.download_and_extract(self.config.data_files)
        splits = []
        for split_name, files in data_files.items():
            if isinstance(files, str):
                files = [files]
            splits.append(datasets.SplitGenerator(
                name=split_name, gen_kwargs={"files": files}))
        return splits

    def _generate_examples(self, files):
        for file in files:
            with open(file, encoding="utf-8") as f:
                # TODO: Validate file types?
                df = pd.read_csv(file)
                df.rename(columns={'Unnamed: 0': 'serial'}, inplace=True)
                for _, row in df.iterrows():
                    yield row['serial'], {
                        "text": row['response'],
                        "far": row['score_spatiotemp'],
                        "obj": row['score_obj']
                    }
