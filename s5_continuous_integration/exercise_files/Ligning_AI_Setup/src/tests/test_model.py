import sys
import os
from logging_helper import LoggerConfigurator
import torch
import re
import pytest

# Get dir of cur script
script_dir = os.path.dirname(__file__)

# Go up to project dir
project_dir = os.path.dirname(script_dir)

# Add data dir to sys path
model_dir = os.path.join(project_dir, 'models')
sys.path.append(model_dir)

# Now import the Model classifier

logger_configurator = LoggerConfigurator('s5_CI')
logger = logger_configurator.get_logger()

print(model_dir)
from model import MNISTModel

model = MNISTModel(in_features=28 * 28, out_features=10)


def test_error_on_wrong_shape():
    with pytest.raises(ValueError, match='Extected input to be a 3D tensor'):
        input_shape = torch.Size([1, 2, 3, 4])
        # Create a test input
        test_input = torch.randn(input_shape)
        # Get model output
        model(test_input)


def test_error_dimensions():
    expected_message = re.escape('Expected each sample to have shape [1,28,28]')
    with pytest.raises(ValueError, match=expected_message):
        input_shape = torch.Size([3, 2, 3])
        # Create a test input
        test_input = torch.randn(input_shape)
        # Get model output
        model(test_input)


def test_model():
    input_shape = torch.Size([1, 28, 28])
    expected_output_shape = torch.Size([1, 10])

    # Create a test input
    test_input = torch.randn(input_shape)
    # Get model output
    model.eval()
    with torch.no_grad():  # Disable gradient computation
        output = model(test_input)
    assert output.shape == expected_output_shape, logger.error(
        f'Output shape mismatch: Expected {expected_output_shape}, got {output.shape}'
    )


if __name__ == '__main__':
    test_error_on_wrong_shape()
    test_error_dimensions()
    # test_model()
