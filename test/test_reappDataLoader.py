import sys
import os
import pytest
from typing import Dict

import numpy as np
import pandas as pd
from datasets import DatasetDict, Dataset

sys.path.insert(0, os.getcwd())
from src.reappDataLoader import encode_data
from src.LDHData import LDHData

@pytest.fixture
def use_src_directory():
    os.chdir(os.path.join(os.getcwd(), "src"))

@pytest.fixture
def ldhdata(use_src_directory):
    return LDHData()


def test_dataframe(ldhdata):
    df = ldhdata.ldh_df
    print(df)

def test_objective_data(ldhdata):
    obj_data = ldhdata.get_obj_data()
    assert obj_data is not None




# class TestDataClass: #TODO: rename for better description and grouping of tests 
#     """Tests methods involved with creating and preprocessing training data.
#     """
#     def test_DatasetNotEncoded(self):
#         # Passes if fails early when trying to pass a non-encoded dataset into create_datasetloader
#         assert True

#     def test_encode_dataset(self):
#         tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-cased')
#         sentences = ["Hello, I'm The Doctor.", "Doctor Who?"]
#         print(tokenizer(sentences))
#         encoded_sentences = tokenizer(sentences)
#         assert encoded_sentences.keys() is not None
        

#     def test_LDH(self, ldh_dataset):
#         """Tests shape of ldh_dataset
#         """
#         # Test shape
#         assert ldh_dataset is not None
#         for split in ldh_dataset.keys():
#             assert len(ldh_dataset[split]) > 0

#     def test_EncodedDatasetsHaveSpecialTokens(self):
#         # Check that CLS and SEP tokens are present in the encoded dataset
#         assert True


# # Tests on the actual data
# # TODO: Find metrics on "good data"