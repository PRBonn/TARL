import click
from os.path import join, dirname, abspath
import subprocess
from pytorch_lightning import Trainer
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
import torch
import yaml

import carl.datasets.datasets as datasets
import carl.models.models as models


@click.command()
### Add your options here
@click.option('--checkpoint',
              '-ckpt',
              type=str,
              help='path to checkpoint file (.ckpt)',
              required=True)
def main(checkpoint):
    cfg = torch.load(checkpoint)['hyper_parameters']
    print(cfg.keys())

    #Load data and model
    data = datasets.StatDataModule(cfg)
    model = models.StatNet.load_from_checkpoint(checkpoint,hparams=cfg)

    #Setup trainer
    trainer = Trainer(logger= False,
        gpus=cfg['train']['n_gpus'])

    # Test!
    trainer.test(model, data.test_dataloader())

if __name__ == "__main__":
    main()
