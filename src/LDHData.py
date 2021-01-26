import os

import numpy as np
import pandas as pd
from datasets import Dataset, DatasetDict

class LDHData:
    def __init__(self):
        # Grab training dataset and save it to training_data
        datadir_train = os.path.join(os.getcwd(),"../input/training")
        files = os.listdir(datadir_train)
        dfs = []
        for file in files:
            if file.endswith(".csv"):
                filename = os.path.join(datadir_train, file)
                dfs.append(pd.read_csv(
                    filename, 
                    header=None, 
                    names=['response', "spatiotemp", "obj"]
                ))
        ldh = pd.concat(dfs, ignore_index=True)
        self.train_data = DatasetDict({
            "spatiotemp" : Dataset.from_pandas(ldh[['response', 'spatiotemp']].rename(columns={"spatiotemp": "score"})),
            "obj" : Dataset.from_pandas(ldh[['response', 'obj']].rename(columns={"spatiotemp": "score"}))
        })

        # Grab testing dataset and set it to eval_data, split by 
        columns = ["addcode", "Subj_ID", "Condition", "TextResponse"]

        datadir_eval = os.path.join(os.getcwd(), "../eval")
        ldhii_far = pd.read_excel(os.path.join(datadir_eval, "Alg_Far_NEW.xlsx" ), usecols=columns)
        ldhii_obj = pd.read_excel(os.path.join(datadir_eval, "Alg_Obj_NEW.xlsx" ), usecols=columns)

        ldhii_far.columns = ["addcode", "subjID", "condition", "response"]
        ldhii_obj.columns = ["addcode", "subjID", "condition", "response"]

        self.eval_data = {
            "spatiotemp": Dataset.from_pandas(ldhii_far),
            "obj": Dataset.from_pandas(ldhii_obj)
        }

