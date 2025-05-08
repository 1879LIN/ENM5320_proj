#!/usr/bin/env python
"""
Train a CNN or H-DNN (Hamiltonian) on the Pendulum dataset.
Usage:
    python run_Pendulum.py --net_type [MODEL NAME] --n_layers [NUMBER OF LAYERS]

Flags:
  --net_type: Network model to use. E.g., CNN_DNN, H1_J1, H1_J2, H2
  --n_layers: Number of layers for the chosen model.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import numpy as np
import argparse
import matplotlib.pyplot as plt

# Import your models here
from integrators.integrators import H1, H2, MS1
from integrators.CNN_DNN import CNN_DNN, train_cnn_dnn
# from your_project import Net_Global  # If you want to use Net_Global

# Placeholder for Pendulum dataset loader
def load_pendulum_data(batch_size, test_batch_size):
    # TODO: Replace this with your actual Pendulum dataset loading logic
    # For now, use random data as a placeholder
    X_train = torch.randn(1000, 1, 28, 28)
    y_train = torch.randint(0, 2, (1000,))
    X_test = torch.randn(200, 1, 28, 28)
    y_test = torch.randint(0, 2, (200,))
    train_loader = torch.utils.data.DataLoader(
        list(zip(X_train, y_train)), batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        list(zip(X_test, y_test)), batch_size=test_batch_size, shuffle=False)
    return train_loader, test_loader

class Net(nn.Module):
    def __init__(self, nf=8, n_layers=4, h=0.5, net_type='H1_J1'):
        super().__init__()
        self.conv1 = nn.Conv2d(1, nf, 3, stride=1, padding=1)
        if net_type == 'MS1':
            self.hamiltonian = MS1(n_layers=n_layers, t_end=h * n_layers, nf=nf)
        elif net_type == 'H1_J1':
            self.hamiltonian = H1(n_layers=n_layers, t_end=h * n_layers, nf=nf, select_j='J1')
        elif net_type == 'H1_J2':
            self.hamiltonian = H1(n_layers=n_layers, t_end=h * n_layers, nf=nf, select_j='J2')
        elif net_type == 'H2':
            self.hamiltonian = H2(n_layers=n_layers, t_end=h * n_layers, nf=nf)
        else:
            raise ValueError(f"{net_type} model is not implemented for Pendulum")
        self.fc_end = nn.Linear(nf, 2)  # 2 classes for placeholder
        self.nf = nf
    def forward(self, x):
        x = self.conv1(x)
        x = x.view(x.size(0), self.nf, -1)
        x = x.mean(dim=2, keepdim=True)
        x = self.hamiltonian(x)
        x = x.view(x.size(0), -1)
        x = self.fc_end(x)
        return F.log_softmax(x, dim=1)
    def encode(self, x):
        x = self.conv1(x)
        x = x.view(x.size(0), self.nf, -1)
        x = x.mean(dim=2, keepdim=True)
        return x

def train(model, device, train_loader, optimizer, epoch, alpha, out):
    model.train()
    for batch_idx, batch in enumerate(train_loader):
        data, target = batch
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0 and out > 0:
            pred = output.argmax(dim=1, keepdim=True)
            correct = pred.eq(target.view_as(pred)).sum().item()
            print(f'\tTrain Epoch: {epoch:2d} [{batch_idx * len(data):5d}/({len(train_loader.dataset)})] Loss: {loss.item():.6f} Accuracy: {correct}/{len(data)}')

def test(model, device, test_loader, out):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for batch in test_loader:
            data, target = batch
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    if out > 0:
        print(f'Test set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({100. * correct / len(test_loader.dataset):.2f}%)')
    return correct

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--net_type', type=str, default='H1_J1')
    parser.add_argument('--n_layers', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--test_batch_size', type=int, default=200)
    args = parser.parse_args()

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    nf = 16 if args.net_type != 'CNN_DNN' else 32
    h = 0.5
    out = 1

    # Load Pendulum data (replace with your actual loader later)
    train_loader, test_loader = load_pendulum_data(args.batch_size, args.test_batch_size)

    # Model selection
    if args.net_type == 'CNN_DNN':
        model = CNN_DNN(nf=nf, n_layers=args.n_layers).to(device)
        train_fn = train_cnn_dnn
    else:
        model = Net(nf=nf, n_layers=args.n_layers, h=h, net_type=args.net_type).to(device)
        train_fn = train

    optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-3)
    scheduler = StepLR(optimizer, step_size=1, gamma=0.8)

    for epoch in range(1, args.epochs + 1):
        if args.net_type == 'CNN_DNN':
            train_fn(model, device, train_loader, optimizer, epoch, out)
        else:
            train_fn(model, device, train_loader, optimizer, epoch, 1e-3, out)
        test(model, device, test_loader, out)
        scheduler.step()

    print("\nTraining complete.")

if __name__ == '__main__':
    main() 