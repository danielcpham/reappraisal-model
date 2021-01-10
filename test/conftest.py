import pytest

@pytest.fixture
def ldh_dataset():
    from datasets import load_dataset
    dataset = load_dataset("./src/LDHDataset.py", data_files = {'train': 'eval/data_train_example.csv', 'test': "eval/data_test_example.csv"})
    return dataset