import datasets
import os

import pandas as pd

#TODO: Use GitLFS to get the files?


class LDHConfig(datasets.BuilderConfig):
  """Builder Config for LDH Data"""
  def __init__(self, **kwargs):
    super(LDHConfig, self).__init__(**kwargs)

class LDHDataset(datasets.GeneratorBasedBuilder):
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
    #TODO: move data online
    data_dir = os.path.join("eval")

    df = pd.read_csv(os.path.join(data_dir, "data_train_example.csv"))
    return [
      datasets.SplitGenerator(
        name="train", gen_kwargs={"filepath": os.path.join(data_dir, "data_train_example.csv")},
      ),
      datasets.SplitGenerator(
        name="validation", gen_kwargs={"filepath": os.path.join(data_dir, "data_test_example.csv")},
      ),
    ]

  def _generate_examples(self, filepath):
    with open(filepath, encoding="utf-8") as f:
      # TODO: Validate file types?
      df = pd.read_csv(filepath)
      df.rename(columns={'Unnamed: 0': 'serial'}, inplace=True)
      for id_, row in df.iterrows():
        yield id_, {
          "text": row['response'],
          "far": row['score_spatiotemp'],
          "obj": row['score_obj']
        }
      