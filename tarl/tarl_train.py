import click
from os.path import join, dirname, abspath
import subprocess
from pytorch_lightning import Trainer
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
import yaml

import tarl.datasets.datasets as datasets
import tarl.models.models as models

@click.command()
### Add your options here
@click.option('--config',
              '-c',
              type=str,
              help='path to the config file (.yaml)',
              default=join(dirname(abspath(__file__)),'config/config.yaml'))
@click.option('--weights',
              '-w',
              type=str,
              help='path to pretrained weights (.ckpt). Use this flag if you just want to load the weights from the checkpoint file without resuming training.',
              default=None)
@click.option('--checkpoint',
              '-ckpt',
              type=str,
              help='path to checkpoint file (.ckpt) to resume training.',
              default=None)
def main(config,weights,checkpoint):
    cfg = yaml.safe_load(open(config))
    cfg['git_commit_version'] = str(subprocess.check_output(
            ['git', 'rev-parse', '--short', 'HEAD']).strip())

    #Load data and model
    data = datasets.data_modules[cfg['data']['dataloader']](cfg)
    # model = models.StatNet(cfg)
    if weights is None:
        model = models.TARLTrainer(cfg, data)
    else:
        print('Loading: ', weights)
        model = models.TARLTrainer.load_from_checkpoint(weights,hparams=cfg)
        model.save_backbone()

    #Add callbacks
    lr_monitor = LearningRateMonitor(logging_interval='step')

    checkpoint_saver = ModelCheckpoint(every_n_epochs=10,
                                 filename=cfg['experiment']['id']+'_{epoch:02d}_{loss:.2f}',
                                 save_top_k=-1,
                                 save_last=True)

    tb_logger = pl_loggers.TensorBoardLogger('experiments/'+cfg['experiment']['id'],
                                             default_hp_metric=False)

    #Setup trainer
    trainer = Trainer(gpus=cfg['train']['n_gpus'],
                      logger=tb_logger,
                      log_every_n_steps=100,
                      resume_from_checkpoint=checkpoint,
                      max_epochs= cfg['train']['max_epoch'],
                      callbacks=[lr_monitor, checkpoint_saver],
                      )#track_grad_norm=True)

    # Train!
    trainer.fit(model, data)

if __name__ == "__main__":
    main()
