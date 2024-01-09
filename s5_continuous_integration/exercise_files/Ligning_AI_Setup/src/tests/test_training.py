import torch
import pytorch_lightning as pl
import sys
import os
from logging_helper import LoggerConfigurator

# Get dir of cur script
script_dir = os.path.dirname(__file__)

# Go up to project dir
project_dir = os.path.dirname(script_dir)
# Add data dir to sys path
model_dir = os.path.join(project_dir, "models")
sys.path.append(model_dir)

from model import MNISTModel

logger_configurator = LoggerConfigurator("s5_CI")
logger = logger_configurator.get_logger()


def test_training_loss_decreases():
    # Initialize the model
    model = MNISTModel(in_features=28 * 28, out_features=10, test=True)

    # Initialize a trainer with limited epochs for testing
    trainer = pl.Trainer(max_epochs=4, logger=False)

    # Train the model
    trainer.fit(model)
    initial_loss = float("inf")

    loss_values = model.metrics["epoch_loss"]
    # Check if the loss decreased
    assert loss_values[-1] < initial_loss, logger.error(
        "Training error did not decrease"
    )


if __name__ == "__main__":
    test_training_loss_decreases()
