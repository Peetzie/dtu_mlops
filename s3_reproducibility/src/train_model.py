import click
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from models import Classifier
from data import FMNIST


@click.group()
def cli():
    pass


@cli.command()
@click.option('--lr', default=1e-3, help='Learning rate to use for training')
@click.option('--epochs', default=30, help='Number of epochs to train for')
def train(lr, epochs):
    """Train a model and save it to disk."""
    train_dataset = FMNIST(train=True)
    trainloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    model = Classifier(in_features=28 * 28, out_features=10)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    steps = 0
    train_losses = []

    for e in range(epochs):
        running_loss = 0
        for images, labels in trainloader:
            optimizer.zero_grad()
            log_ps = model(images)
            loss = criterion(log_ps, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            train_losses.append(running_loss / len(trainloader))
        print(
            f'Epoch: {e+1}/{epochs}',
            f'Training Loss: {running_loss/len(trainloader):.3f}',
        )


if __name__ == '__main__':
    cli()
