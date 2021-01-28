import os
import sys
import pytest

import pandas as pd
from datasets import Dataset
from transformers import DistilBertTokenizer

sys.path.insert(0, os.getcwd())
from src.ReappModel import ReappModel


@pytest.fixture
def tokenizer():
    tok = DistilBertTokenizer.from_pretrained("distilbert-base-cased")
    tok.model_input_names.append("score")


@pytest.fixture
def train_example():
    # dataset model of 1 
    # Get tokenizer
    # Tokenize the thing
    # return the tokenized dataset
    example = Dataset.from_dict({
        "response": ["This might be the coolest thing in the world!"] * 16,
        "score": [6.0] * 16
    })
    return example


def test_train_example(train_example, tokenizer):
    encoded = train_example.map(lambda example: tokenizer(
        example['response'],
        add_special_tokens=True,
        padding=True,
        truncation=True), batched=True, batch_size=16)
    encoded.set_format(type='torch', columns=['attention_mask', 'input_ids', 'score'], output_all_columns=True)
    assert encoded is not None
    print(encoded)

    model = ReappModel()
    outputs = model(input_ids=encoded['input_ids'], attention_mask=encoded['attention_mask'], scores=encoded['score'])
    print(outputs)
    
