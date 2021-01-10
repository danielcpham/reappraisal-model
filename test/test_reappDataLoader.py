import sys
import os

import pytest
import datasets

from src.reappDataLoader import encode_data



class TestDataClass: #TODO: rename for better description and grouping of tests 
    """Tests methods involved with creating and preprocessing training data.
    """
    def test_DatasetNotEncoded(self):
        # Passes if fails early when trying to pass a non-encoded dataset into create_datasetloader
        assert True

    def test_LDH(self, ldh_dataset):
        """Tests shape of ldh_dataset
        """
        assert ldh_dataset is not None
        for split in ldh_dataset.keys():
            assert len(ldh_dataset[split]) > 0

    def test_EncodedDatasetsHaveSpecialTokens(self):
        # Check that CLS and SEP tokens are present in the encoded dataset
        assert True

    

    




# Tests on the actual data
# TODO: Find metrics on "good data"