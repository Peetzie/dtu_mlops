import click
from models import MNISTModel
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
import os
import sys
import torch
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
import matplotlib.pyplot as plt
from tests.logging_helper import LoggerConfigurator

# SCRIPT DIR
script_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
model = MNISTModel()  # LigningModule


@click.group()
def cli():
    pass


logger_configurator = LoggerConfigurator('s5')
logger = logger_configurator.get_logger()


@cli.command()
def train():
    logger.info('Beginning training')
    """Train a model and save it to disk."""
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(script_dir, 'Models/'),
        monitor='train_loss',
        mode='min',
    )
    early_stopping_callback = EarlyStopping(
        monitor='train_loss', patience=3, verbose=True, mode='min'
    )

    trainer = Trainer(
        logger=WandbLogger(project='dtu_mlops'),
        max_epochs=10,
        callbacks=[checkpoint_callback, early_stopping_callback],
    )

    trainer.fit(model=model)

    def report():
        logger.info('Finished training.. Beginning reporting')
        preds, target = [], []
        loader = model.train_dataloader()
        for batch in loader:
            x, y = batch
            probs = model(x)
            preds.append(probs.argmax(dim=-1))
            target.append(y.detach())
        target = torch.cat(target, dim=0)
        preds = torch.cat(preds, dim=0)
        report_ = classification_report(target, preds)
        with open(script_dir + 'classification_report.txt', 'w') as outfile:
            outfile.write(report_)
        confmat = confusion_matrix(target, preds)
        confusion_display = ConfusionMatrixDisplay(confusion_matrix=confmat)
        confusion_display.plot(
            cmap='viridis', values_format='.4g'
        )  # Customize cmap and format as needed
        plt.savefig(script_dir + 'confusion_matrix.png')  # Save the plot
        logger.info('Finished reporting')

    report()


if __name__ == '__main__':
    train()
