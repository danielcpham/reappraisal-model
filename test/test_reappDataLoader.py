import sys
import os

import pytest

from src.runUtils import run_epoch

class TestDataClass:
    """Tests methods involved with creating and preprocessing training data.
    """

def test_DatasetNotEncoded():
  # Passes if fails early when trying to pass a non-encoded dataset into create_datasetloader
  assert True

def test_encoder():
  pass

# Tests on the actual data
# TODO: Find metrics on "good data"