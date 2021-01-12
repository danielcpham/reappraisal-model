from typing import Tuple
from collections import defaultdict as ddict

import numpy as np

import torch
from torch import nn
from torch.utils.data import DataLoader

# https://pytorch.org/tutorials/beginner/hyperparameter_tuning_tutorial.html


def train_model(
        model,
        train_dl: DataLoader,
        loss_fn, 
        optimizer, 
        scheduler, 
        device = None) -> Tuple[float, float]:
    """
    Runs a training pass.
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
    if not model.training:
        model = model.train()
    losses = []
    correct_predictions = 0
    for data in train_dl:
        input_ids = data['input_ids']
        attention_mask = data['attention_mask']
        labels = data['label']

        # call forward on the model 
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        _, preds = torch.max(outputs, dim=1)
        loss = loss_fn(outputs, labels)
        correct_predictions += torch.sum(preds == labels)
        losses.append(loss.item())

        # Propagate the updates to autograd backwards. 
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()  # clear out gradients so the model doesn't reuse them in the next training.

    return correct_predictions.double() / len(train_dl.dataset), np.mean(losses)


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

def run_epoch(epoch:int, model, train_dataloader : DataLoader, val_dataloader: DataLoader, loss_fn, optimizer, device, scheduler):
    print(f'epoch {epoch+1}')
    print('-' * 10)
    train_acc, train_loss = train_model(
        model,
        train_dataloader,
        loss_fn,
        optimizer,
        device,
        scheduler
    )

    print(f'Loss: {train_loss}, Accuracy: {train_acc}')

    val_acc, val_loss = eval_model(
        model,
        val_dataloader,
        loss_fn,
        device
    )
    print(f'Val: Loss: {val_loss}, Accuracy: {val_acc}')
    print()




