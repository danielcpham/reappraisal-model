from typing import Tuple

import numpy as np
from datasets import DataLoader

import torch
from torch import nn, optim

def run_epoch(
  model:Model, 
  dataloader: DataLoader, 
  loss_fn, optimizer,scheduler, n_examples, device=None) -> Tuple[float, float]:
    model = model.train()
    losses = []
    correct_predictions = 0

    for data in dataloader:
        input_ids = data['input_ids']
        attention_mask = data['attention_mask']
        labels = data['label']

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)

        _, preds = torch.max(outputs, dim=1)
        loss = loss_fn(outputs, labels)

        correct_predictions += torch.sum(preds == labels)
        losses.append(loss.item())

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad() # clear out gradients

    return correct_predictions.double() / n_examples, np.mean(losses)

def eval_model(model, dataloader, loss_fn, n_examples, device=None):
    model = model.eval()

    with torch.no_grad():
        losses = []
    correct_predictions = 0

    for data in dataloader:
        input_ids = data['input_ids']
        attention_mask = data['attention_mask']
        labels = data['label']

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)

        _, preds = torch.max(outputs, dim=1)
        loss = loss_fn(outputs, labels)

        correct_predictions += torch.sum(preds == labels)
        losses.append(loss.item())
    
    return correct_predictions.double() / n_examples, np.mean(losses)