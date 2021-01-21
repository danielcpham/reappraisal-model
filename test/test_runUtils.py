import os
import sys

import pytest

sys.path.insert(0, os.getcwd())
from src.runUtils import train_model, eval_model, run_epoch

def test_trainmodel():
    """
    - Test that the correct parameters were upgraded
    - Test correct running of a forward pass.
    - Test proper backpropagation
    - Test that zero_grad was called
    - Test that the original model was updated and not copied.
    - Test that the gradiants were normalized
    - Test correct calculation of accuracy and loss (use random seed)
    """
    assert True
