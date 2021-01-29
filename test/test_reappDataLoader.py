import sys
import os
import pytest
from typing import Dict, List

import numpy as np
import pandas as pd
from transformers import DistilBertTokenizer
from datasets import DatasetDict, Dataset

sys.path.insert(0, os.getcwd())
from src.LDHData import LDHData

@pytest.fixture
def ldhdata():
    return LDHData(tokenizer=DistilBertTokenizer.from_pretrained('distilbert-base-cased'))


def test_encode_eval_data(ldhdata):
    far = ldhdata.eval_far_data
    # obj = ldhdata.eval_obj_data
    from nltk.tokenize import sent_tokenize
    def expand_response(str):
        sentences = sent_tokenize(str)
        return sentences
    new_responses = far['response'].map(lambda resp: expand_response(resp)) \
        .apply(pd.Series)\
        .unstack()\
        .reset_index()\
        .drop('level_1', axis=1)\
        .dropna()
    print(new_responses)
    # ldhdata.encode_eval_data(far)
