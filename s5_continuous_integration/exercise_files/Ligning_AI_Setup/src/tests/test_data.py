import sys
import os
from logging_helper import LoggerConfigurator
import torch
import pytest

# Get dir of cur script
script_dir = os.path.dirname(__file__)

# Go up to project dir
project_dir = os.path.dirname(script_dir)

# Add data dir to sys path
data_dir = os.path.join(project_dir, 'data')
sys.path.append(data_dir)

# Now import FMNIST dataloader
from make_dataset import FMNIST


logger_configurator = LoggerConfigurator('s5_CI')
logger = logger_configurator.get_logger()
train_dataset = FMNIST(train=True)
test_dataset = FMNIST(train=False)


@pytest.mark.skipif(not os.path.exists('data/raw'), reason='Data files not found')
def test_data_pytest():
    test_data()


def test_data():
    N_train = 30000  # First subset
    N_test = 5000
    assert len(train_dataset) == N_train and len(test_dataset) == N_test

    def check_shape(dataset, expected_shape_img, expected_shape_label=None):
        for data, labels in dataset:
            assert data.shape == expected_shape_img, logger.error(
                f'Data point shape mismatch: Expected {expected_shape_img}, got {data.shape}'
            )

            if not expected_shape_label:
                assert labels.shape == expected_shape_2, logger.error(
                    f'Label point shape mismatch: Expected {expected_shape_label}, got {labels.shape}'
                )

    expected_shape_1 = torch.Size([1, 28, 28])
    expected_shape_2 = torch.Size(
        [
            784,
        ]
    )
    # Choose the expected shape based on your dataset format
    check_shape(
        train_dataset, expected_shape_1, expected_shape_2
    )  # or expected_shape_2

    def check_labels(dataset):
        for data, labels in dataset:
            assert labels < 10, logger.error(
                f'Label value mismatch: Expected label < 10, got {labels}'
            )

    check_labels(train_dataset)


if __name__ == '__main__':
    test_data_pytest()
