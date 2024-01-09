from pytorch_lightning import LightningModule
from pytorch_lightning.utilities.types import OptimizerLRScheduler
from torch import nn, optim
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import torch.nn.functional as F
import numpy as np
import sys
import torch
import wandb
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import os

# Get dir of cur script
script_dir = os.path.dirname(__file__)

# Go up to project dir
project_dir = os.path.dirname(script_dir)

# Add data dir to sys path
data_dir = os.path.join(project_dir, 'data')
sys.path.append(data_dir)

# Now import FMNIST dataloader
from make_dataset import FMNIST


class MNISTModel(LightningModule):
    """Basic neural network class.

    Args:
        in_features: number of input features
        out_features: number of output features

    """

    def __init__(self, test=False, batch_size=64) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 64, 3),
            nn.LeakyReLU(),
            nn.Conv2d(64, 32, 3),
            nn.LeakyReLU(),
            nn.Conv2d(32, 16, 3),
            nn.LeakyReLU(),
            nn.Conv2d(16, 8, 3),
            nn.LeakyReLU(),
        )
        self.batch_size = batch_size
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(8 * 20 * 20, 128), nn.Dropout(), nn.Linear(128, 10)
        )
        self.loss_function = nn.CrossEntropyLoss()

        self.test = test
        if self.test:
            self.metrics = {'loss': [], 'epoch_loss': []}  # Testing metrics

    def forward(self, x: torch.Tensor):
        if x.ndim != 4:  # Bach, width, height (3D tesor)
            raise ValueError('Extected input to be a 4D tensor')
        if x.shape[0] != self.batch_size or x.shape[2] != 28 or x.shape[3] != 28:
            raise ValueError('Expected each sample to have shape [64,1,28,28]')
        return self.classifier(self.conv(x))

    def training_step(self, batch, batch_idx):
        data, target = batch
        preds = self(data)
        loss = self.loss_function(preds, target.squeeze())
        acc = (target.squeeze() == preds.argmax(dim=-1)).float().mean()
        self.log('train_loss', loss)
        self.log('train_acc', acc)
        if self.logger:
            self.logger.experiment.log(
                {'logits': wandb.Histogram(preds.cpu().detach())}
            )
        if self.test:
            self.metrics['loss'].append(loss.item())
        return loss

    def on_train_epoch_end(self):
        if self.test:
            avg = np.mean(self.metrics['loss'])
            self.metrics['epoch_loss'].append(avg)
            self.metrics['loss'] = []

    def validation_step(self, batch, batch_idx):
        data, target = batch
        preds = self(data)
        loss = self.criterion(preds, target)
        acc = (target == preds.argmax(dim=-1)).float().mean()
        self.log('val_loss', loss)
        self.log('val_acc', acc)
        return {'val_loss': loss, 'val_acc': acc}

    def configure_optimizers(self) -> OptimizerLRScheduler:
        return optim.Adam(self.parameters(), lr=1e-4)

    def train_dataloader(self):
        return DataLoader(FMNIST(train=True), batch_size=self.batch_size, shuffle=True)

    def test_dataloader(self):
        return DataLoader(FMNIST(train=False), batch_size=self.batch_size, shuffle=True)


if __name__ == '__main__':
    from pytorch_lightning import Trainer

    trainer = Trainer(
        precision='32-true',
        profiler='simple',
        max_epochs=10,
        logger=pl.loggers.WandbLogger(project='dtu_mlops'),
        callbacks=[EarlyStopping(monitor='train_loss', mode='min')],
    )
    model = MNISTModel(test=False)
    trainer.fit(model, model.train_dataloader())
