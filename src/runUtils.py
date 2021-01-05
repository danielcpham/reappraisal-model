from typing import Tuple

import numpy as np
from datasets import DataLoader

import torch
from torch import nn, Optimizer, Scheduler

class Run:
    #TODO: put an epoch run loop into the class. 
    def __init__(self):
        pass
    

def run_epoch(
        model: Model,
        dataloader: DataLoader,
        loss_fn, 
        optimizer: Optimizer, 
        scheduler: Scheduler, 
        n_examples: int, device = None) -> Tuple[float, float]:
    """
    Runs a training epoch.
    Args:
        model (Model): [description]
        dataloader (DataLoader): [description]
        loss_fn ([type]): [description]
        optimizer ([type]): [description]
        scheduler ([type]): [description]
        n_examples ([type]): [description]
        device ([type], optional): [description]. Defaults to None.

    Returns:
        Tuple[float, float]: 
            - Accuracy of the model after training
            - Avg. Loss of the model after training
    """
    # Put the model in training mode.
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
        optimizer.zero_grad()  # clear out gradients

    return correct_predictions.double() / n_examples, np.mean(losses)


def eval_model(model, dataloader, loss_fn, n_examples, device=None):
    # Put the model in eval mode
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
