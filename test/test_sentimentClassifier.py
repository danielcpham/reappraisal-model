import os
import sys
import pytest

sys.path.insert(0, os.getcwd())
from src.ReappModel import ReappModel

class TestModel:
    def test_reappConfig(self):
        model = ReappModel()
        print(model)
        assert True
