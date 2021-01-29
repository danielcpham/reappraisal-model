import sys
import os
import pytest
from typing import Dict, List

import numpy as np
import pandas as pd
from transformers import DistilBertTokenizer
from datasets import DatasetDict, Dataset

sys.path.insert(0, os.getcwd())
from src.LDHData import LDHData, _expand_response

@pytest.fixture
def ldhdata() -> LDHData:
    return LDHData(tokenizer=DistilBertTokenizer.from_pretrained('distilbert-base-cased'))
    

def test_encode_eval_data(ldhdata: LDHData):
    far : pd.DataFrame = ldhdata.get_eval_far_data(encoded=False)
    # obj = ldhdata.eval_obj_data   
    
    # ldhdata.encode_eval_data(far)


