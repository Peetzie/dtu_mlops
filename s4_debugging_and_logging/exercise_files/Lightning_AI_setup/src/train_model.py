import click
from models import MNISTModel
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
import os
import sys

# SCRIPT DIR
script_dir = os.path.dirname(os.path.abspath(sys.argv[0]))


@click.group()
def cli():
    pass


@cli.command()
def train():
    """Train a model and save it to disk."""
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(script_dir, 'Models/'),
        monitor='val_loss',
        mode='min',
    )
    early_stopping_callback = EarlyStopping(
        monitor='val_loss', patience=3, verbose=True, mode='min'
    )

    model = MNISTModel(in_features=28 * 28, out_features=10)  # LigningModule
    trainer = Trainer(
        logger=WandbLogger(project='dtu_mlops'),
        max_epochs=30,
        callbacks=[checkpoint_callback, early_stopping_callback],
    )

    trainer.fit(model=model)


if __name__ == '__main__':
    cli()
