# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/Trainers.ipynb (unless otherwise specified).

__all__ = ['kfold_train', 'default_config', 'train_tune', 'parser', 'callback_tuner', 'default_tune_config',
           'upload_file']

# Cell
#export
import datetime
import os
import tempfile

import torch
import pandas as pd
import pytorch_lightning as lit
import wandb
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger, WandbLogger

from .lightningreapp import LightningReapp

# Default configuration for hyperparameters
default_config = {
    'lr': 1e-3,
    'hidden_layer_size': 50
    }


def kfold_train(k: int, ldhdata, strat, **trainer_kwargs) -> None:
    """Fits a LightningReapp instance with k-fold cross-validation.
    Args:
        k (int):
        ldhdata : See `reappraisalmodel.ldhdata.LDHDataModule`
    """
    all_metrics = []


    
    
    max_epochs = trainer_kwargs.pop('max_epochs', 20)
    gpus = trainer_kwargs.pop('gpus', 1 if torch.cuda.is_available() else None)

    today = datetime.datetime.today().strftime('%Y%m%d_%H%M%S')

    with tempfile.TemporaryDirectory() as tempdir:
        print(f'Created temporary directory: {tempdir}')

        for i in range(k):
            # Select the dataloaders for the given split.
            split = i
            train_dl = ldhdata.get_train_dataloader(split)
            val_dl = ldhdata.get_val_dataloader(split)
            
            save_dir='lightning_logs'
            name=f"reappmodel_{strat}_{today}"
            version=f'{split}'
            prefix=split
            
            # Loggers
            logger = TensorBoardLogger(
                save_dir=save_dir,
                name=name,
                version=version,
                prefix=prefix
            )
            
            csv_logger = CSVLogger(
                save_dir=save_dir,
                name=name,
                version=version,
                prefix=prefix
            )
            
            #Checkpoints
            early_stop_checkpoint = EarlyStopping(
                monitor='val_loss',
                mode='min',
                min_delta=0.001,
                patience=3,
                verbose=False
            )
            
            callback_checkpoint = ModelCheckpoint(
                monitor='val_loss',
                mode='min',
                dirpath=os.path.join(tempdir, name), 
                filename= f'{split}_'+'{epoch:02d}-{val_loss:.02f}',
                verbose=False,
                save_last=False,
                save_top_k=1,
                save_weights_only=False,
            )
                        
            model = LightningReapp(default_config)
            trainer = lit.Trainer(
                benchmark=True,
                logger = [logger, csv_logger],
                gpus = gpus,
                gradient_clip_val=1.0,
                max_epochs=max_epochs,
                terminate_on_nan=True,
                weights_summary=None,
                precision=16,
                callbacks=[callback_checkpoint, early_stop_checkpoint],
                **trainer_kwargs)
            print(f"Training on split {i}")
            trainer.fit(model, train_dl, val_dl)
            all_metrics.append({
                'metrics': trainer.logged_metrics,
                'checkpoint': callback_checkpoint.best_model_path,
                'num_epochs': trainer.current_epoch
            })
     
        outputs = []
        for split in all_metrics:
            val_loss = split['metrics']['val_loss'].item()
            train_loss = split['metrics']['train_loss'].item()
            num_epochs = split['num_epochs']
            r2score = split['metrics']['r2score']
            explained_variance = split['metrics']['explained_var']

            ckpt_path = split['checkpoint']
            filename = os.path.split(ckpt_path)[-1]
            
            upload_result = upload_file(ckpt_path, 'ldhdata', f'{strat}/{split}-{str(today)}-{filename}')
            print(f"Successful {filename} to s3: {upload_result}")

            row = {
                'val_loss': val_loss,
                'train_loss': train_loss,
                'num_epochs': num_epochs,
                'r2score': r2score,
                'explained_var': explained_variance
            }
            print(row)
            outputs.append(row)
            
    return outputs

# Cell
from functools import partial
from argparse import ArgumentParser

import torch
import pytorch_lightning as lit
from ray.tune.integration.pytorch_lightning import TuneReportCallback
from ray import tune

from .lightningreapp import LightningReapp

parser = ArgumentParser()


callback_tuner = TuneReportCallback(
    {
        "loss": "val_loss",
        # "mean_accuracy": "val_accuracy"
    },
    on="validation_end",
)

default_tune_config = {
    "lr": tune.loguniform(1e-4, 1e-1), # loguniform samples by magnitude
    "hidden_layer_size": tune.quniform(10, 50, 1)
}

### TUNING HYPERPARAMETERS
def train_tune(config, **tuner_kwargs):
    model = LightningReapp(config)

    max_epochs = tuner_kwargs.get('max_epochs', 10)
    trainer = lit.Trainer(
        num_folds=3,
        fast_dev_run=1,
        max_epochs=max_epochs,
        gpus= 1 if torch.cuda.is_available() else None,
        progress_bar_refresh_rate=30,
        callbacks=[callback_tuner],
    )
    trainer.fit(model, ldhdata)


# tune.run(train_tune, config=default_tune_config, num_samples=2)

# Cell
import boto3
from botocore.exceptions import ClientError

def upload_file(file_name, bucket, object_name=None):
    """Upload a file to an S3 bucket

    :param file_name: File to upload
    :param bucket: Bucket to upload to
    :param object_name: S3 object name. If not specified then file_name is used
    :return: True if file was uploaded, else False
    """

    # If S3 object_name was not specified, use file_name
    if object_name is None:
        object_name = file_name

    # Upload the file
    s3_client = boto3.client('s3')
    try:
        response = s3_client.upload_file(file_name, bucket, object_name)
    except ClientError as e:
        logging.error(e)
        return False
    return True