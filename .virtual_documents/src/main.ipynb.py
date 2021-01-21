# get_ipython().run_line_magic("%capture", "")
# get_ipython().getoutput("pip install wandb -qqq")
# import wandb
# get_ipython().getoutput("wandb login")

## Sample code for tracking model training runs in wandb 
# see: https://colab.research.google.com/github/wandb/examples/blob/master/colabs/intro/Intro_to_Weights_get_ipython().run_line_magic("26_Biases.ipynb#scrollTo=-VE3MabfZAcx", "")
# import math
# import random

# # 1️⃣ Start a new run, tracking config metadata
# wandb.init(project="test-drive", config={
#     "learning_rate": 0.02,
#     "dropout": 0.2,
#     "architecture": "CNN",
#     "dataset": "CIFAR-100",
# })
# config = wandb.config

# # Simulating a training or evaluation loop
# for x in range(50):
#     acc = math.log(1 + x + random.random() * config.learning_rate) + random.random()
#     loss = 10 - math.log(1 + x + random.random() + config.learning_rate * x) + random.random()
#     # 2️⃣ Log metrics from your script to W&B
#     wandb.log({"acc":acc, "loss":loss})

# wandb.finish()


# TODO: Add Open in Colab Button
# TODO: Write scripts for running as CLI in pipfile
# TODO: hyperparameter search


import os
import numpy as np
import pandas as pd
import torch


from datasets import ReadInstruction

# Enable GPU usage, if we can.
if torch.cuda.is_available():
    print("Enabling GPU usage")
    device = torch.device("cuda:0")
    IS_GPU = True
else:
    print("No GPU available, running on CPU")
    device = torch.device("cpu") # Note: macOS incompatible with NVIDIA GPUs
    IS_GPU = False
    
# Constants and environment setup
# TODO: Set up env files for dev and "prod"
#Casing can matter for sentiment analysis ("bad" vs. "BAD")
PRETRAINED_MODEL_NAME = 'distilbert-base-cased'



from sklearn.model_selection import train_test_split

# Read LDH Data into a Pandas DataFrame.
files = os.listdir("../eval")
dfs = []
for file in files:
    if file.endswith(".csv"):
        dfs.append(pd.read_csv(os.path.join("../eval", file), header=None, names=['response', "spatiotemp", "obj"]))
ldh = pd.concat(dfs, ignore_index=True)


from collections import Counter, defaultdict
from nltk.tokenize import sent_tokenize, word_tokenize

# Word tokenizer and sentence tokenizer with NLTK
resp_lengths = Counter() # Counts
length_scores_spatiotemp = defaultdict(list)
length_scores_obj = defaultdict(list)
for row in ldh.itertuples():
    if row.Index == 0:
        continue
    response = row.response
    score_spat = row.spatiotemp
    score_obj = row.obj
    len_response = len(word_tokenize(response))
    resp_lengths[len_sentence] += 1
    length_scores_spatiotemp[len_sentence].append(score_spat)
    length_scores_obj[len_sentence].append(score_obj)
#     len_sentence = len(word_tokenize(sentence))
#     resp_lengths[len_sentence] += 1
print(resp_lengths)
print(length_scores_spatiotemp)
print(length_scores_obj)

        


# Split LDH Data into a training dataset and a evaluation dataset.
train_ldh, eval_ldh = train_test_split(ldh, test_size=0.15) # shuffle
train_ldh_ds = Dataset.from_pandas(train_ldh)
eval_ldh_ds = Dataset.from_pandas(eval_ldh)
# TODO: Convert to DatasetDict


from datasets import Dataset, load_dataset
# For testing on a CPU, just grab the first few.

if IS_GPU:
    splits = [ReadInstruction('train'), ReadInstruction('test')]
else:
    splits = [ReadInstruction('train', to=256, unit="abs"), ReadInstruction('test', to=64, unit="abs")]

train_ds, eval_ds = load_dataset('imdb', split=splits)

# Split training data into model training and model validation
train_val_ds = train_ds.train_test_split(test_size=0.15)





from torch import nn, optim
from torch.utils.data import DataLoader
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer, AdamW, get_linear_schedule_with_warmup

# Tokenize the datasets.
tokenizer = DistilBertTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)
encoded_ds= train_val_ds.map(
    lambda batch: tokenizer(
        batch['text'],
        add_special_tokens=True,
        padding=True,
        truncation=True), 
    batched=True, batch_size=16, remove_columns=['text'])

# Reformat the dataset to PyTorch tensors.
encoded_ds.set_format(type='torch')
encoded_ds.column_names, encoded_ds.shape


from transformers import DistilBertForSequenceClassification

# Define model.
class TestModel(nn.Module):
    def __init__(self):
        """Initializes the NN wrapper for LDH classification.
        """
        super(TestModel, self).__init__()
        self.inner_model = DistilBertForSequenceClassification.from_pretrained(PRETRAINED_MODEL_NAME)

    def forward(self, input_ids, attention_mask, **kwargs):
        """Called to run a single pass through the training data.
        """
        # Pass the inputs directly into the inner model
        # TODO: Do stuff after we get the forward models
        return self.inner_model.forward(input_ids=input_ids, attention_mask=attention_mask, **kwargs)



from transformers import TrainingArguments, Trainer, DistilBertModel

num_train_epochs = 3 if IS_GPU else 1

# Define the parameters under which the model will be trained.
# By default, uses an AdamW optimizer w/ linear warmup.

training_args = TrainingArguments(
    output_dir='./results',          # output directory
    num_train_epochs=num_train_epochs,
    per_device_train_batch_size=16,  # batch size per device during training
    per_device_eval_batch_size=64,   # batch size for evaluation
    weight_decay=0.01,               # strength of weight decay
    logging_dir='./logs',            # directory for storing logs
)

encoded_train = encoded_ds['train']
encoded_test  = encoded_ds['test']

# Create the training model.
# TODO: Suppress initialization errors.
model = TestModel()

# HyperParameter search depending on the model.

trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    args=training_args,                  
    train_dataset=encoded_train,      
    eval_dataset=encoded_test         
)


trainer.train()


# Model Evaluation: Parse the TrainOutput Object 
