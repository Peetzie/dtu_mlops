from pytorch_lightning import LightningModule
from pytorch_lightning.utilities.types import OptimizerLRScheduler
from torch import nn, optim
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import torch.nn.functional as F
import numpy as np
import sys
import torch
import os

# Get dir of cur script
script_dir = os.path.dirname(__file__)

# Go up to project dir
project_dir = os.path.dirname(script_dir)

# Add data dir to sys path
data_dir = os.path.join(project_dir, "data")
sys.path.append(data_dir)

# Now import FMNIST dataloader
from make_dataset import FMNIST


class MNISTModel(LightningModule):
    def __init__(self, in_features: int, out_features: int, test=False) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_features, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, out_features)
        self.dropout = nn.Dropout(p=0.2)
        self.criterion = nn.CrossEntropyLoss()
        self.test = test
        if self.test:
            self.metrics = {"loss": [], "epoch_loss": []}  # Testing metrics

    def forward(self, x: torch.Tensor):
        if x.ndim != 3:  # Bach, width, height (3D tesor)
            raise ValueError("Extected input to be a 3D tensor")
        if x.shape[0] != 1 or x.shape[1] != 28 or x.shape[2] != 28:
            raise ValueError("Expected each sample to have shape [1,28,28]")
        # make sure input tensor is flattened
        x = x.view(x.shape[0], -1)

        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.dropout(F.relu(self.fc3(x)))
        x = F.log_softmax(self.fc4(x), dim=1)

        return x

    def training_step(self, batch, batch_idx):
        data, target = batch
        preds = self(data)
        loss = self.criterion(preds, target)
        acc = (target == preds.argmax(dim=-1)).float().mean()
        self.log("train_loss", loss)
        self.log("train_acc", acc)
        if self.logger:
            self.logger.experiment.log({"logits": (preds)})
        if self.test:
            self.metrics["loss"].append(loss.item())
        return loss

    def on_train_epoch_end(self):
        if self.test:
            avg = np.mean(self.metrics["loss"])
            self.metrics["epoch_loss"].append(avg)
            self.metrics["loss"] = []

    def validation_step(self, batch, batch_idx):
        data, target = batch
        preds = self(data)
        loss = self.criterion(preds, target)
        acc = (target == preds.argmax(dim=-1)).float().mean()
        self.log("val_loss", loss)
        self.log("val_acc", acc)
        return {"val_loss": loss, "val_acc": acc}

    def configure_optimizers(self) -> OptimizerLRScheduler:
        return optim.Adam(self.parameters(), lr=1e-2)

    def train_dataloader(self):
        train_dataset = FMNIST(train=True)
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        return train_loader

    def val_dataloader(self):
        validation_dataset = FMNIST(train=False)
        validation_loader = DataLoader(validation_dataset, batch_size=64, shuffle=False)
        return validation_loader
