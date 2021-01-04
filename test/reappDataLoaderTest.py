import pytest

from reappDataLoader import create_datasetloader

def test_DatasetNotEncoded():
  # Passes if fails early when trying to pass a non-encoded dataset into create_datasetloader
  assert create_datasetloader() == None

def test_encoder():
  pass