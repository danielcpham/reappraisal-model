import os
from typing import Dict

import numpy as np
import pandas as pd
from datasets import Dataset, Features

class LDHData:
    def __init__(self):
        # Read training data into a single dataframe
        datadir_train = os.path.join(os.getcwd(),"../input/training")
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

        self.ldh_df = ldh
        # Split training dataframe into faraway and obj datasets
        # TODO: rename columns using Dataset features
        ldh_train_far =  Dataset.from_pandas(ldh[['response', 'spatiotemp']])
        ldh_train_obj = Dataset.from_pandas(ldh[['response', 'obj']])

        ldh_train_far.rename_column_('spatiotemp', 'score')
        ldh_train_obj.rename_column_('obj', 'score')

        # Read evaluation data and save as dataframes
        datadir_eval = os.path.join(os.getcwd(), "../eval")
        columns = ["addcode", "Subj_ID", "Condition", "TextResponse"]
        ldh_eval_far = pd.read_excel(os.path.join(datadir_eval, "Alg_Far_NEW.xlsx" ), usecols=columns)
        ldh_eval_obj = pd.read_excel(os.path.join(datadir_eval, "Alg_Obj_NEW.xlsx" ), usecols=columns)

        # Convert datasets into far and obj datasets with training and testing data splits
        self.ldh_far_ds = {
            'train': ldh_train_far,
            'eval': ldh_eval_far
        }
        
        self.ldh_obj_ds = {
            'train': ldh_train_obj,
            'eval': ldh_eval_obj
        }

    def get_spatiotemp_data(self) -> Dict[str, Dataset]:
        """Retrieves the spatiotemporal distancing dataset.
        """
        return self.ldh_far_ds
        
    def get_obj_data(self) -> Dict[str, Dataset]:
        """Retrieves the objective distancing dataset.
        """
        return self.ldh_obj_ds



    