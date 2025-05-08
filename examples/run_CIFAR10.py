#!/usr/bin/env python
"""
Train a H-DNN on CIFAR-10 dataset.
Author: Clara Galimberti (clara.galimberti@epfl.ch)
Usage:
python run_CIFAR10.py     --net_type      [MODEL NAME]            \
                          --n_layers      [NUMBER OF LAYERS]      \
                          --gpu           [GPU ID]
Flags:
  --net_type: Network model to use. Available options are: MS1, H1_J1, H1_J2.
  --n_layers: Number of layers for the chosen the model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import numpy as np
import matplotlib.pyplot as plt

from integrators.integrators import MS1, H1, H2, H2_Global, H1_Global
from regularization.regularization import regularization
import argparse


class Net(nn.Module):
    def __init__(self, nf=32, n_layers=4, h=0.5, net_type='H1_J1'):
        super(Net, self).__init__()
        # First conv layer with more filters for CIFAR-10
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=nf, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(nf)  # Added batch normalization
        
        if net_type == 'MS1':
            self.hamiltonian = MS1(n_layers=n_layers, t_end=h * n_layers, nf=nf)
        elif net_type == 'H1_J1':
            self.hamiltonian = H1(n_layers=n_layers, t_end=h * n_layers, nf=nf, select_j='J1')
        elif net_type == 'H1_J2':
            self.hamiltonian = H1(n_layers=n_layers, t_end=h * n_layers, nf=nf, select_j='J2')
        elif net_type == 'H2':
            self.hamiltonian = H2(n_layers=n_layers, t_end=h * n_layers, nf=nf)
        else:
            raise ValueError("%s model is not yet implemented" % net_type)
        
        self.fc_end = nn.Linear(nf*32*32, 10)  # Adjusted for CIFAR-10 image size
        self.nf = nf

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)  # Apply batch normalization
        x = F.relu(x)    # Added ReLU activation
        x = x.view(x.size(0), self.nf, -1)
        x = self.hamiltonian(x)
        x = x.reshape(-1, self.nf*32*32)
        x = self.fc_end(x)
        output = F.log_softmax(x, dim=1)
        return output


def get_intermediate_states(model, Y0):
    """Track intermediate states in the Hamiltonian evolution."""
    Y0.requires_grad_(True)
    Y_out = [Y0]

    for j in range(model.hamiltonian.n_layers):
        Y = model.hamiltonian(Y_out[j], ini=j, end=j+1)
        Y.retain_grad()
        Y_out.append(Y)

    return Y_out


def train(model, device, train_loader, optimizer, epoch, alpha, out):
    model.train()

    # Initialize gradient history tracker once
    if not hasattr(model, 'grad_norm_history'):
        model.grad_norm_history = {}  # Changed from list to dict

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        K = model.hamiltonian.getK()
        b = model.hamiltonian.getb()
        for j in range(int(model.hamiltonian.n_layers) - 1):
            loss = loss + regularization(alpha, h, K, b)
        loss.backward()

        # Log gradient norms
        for name, param in model.named_parameters():
            if param.grad is not None:
                gnorm = param.grad.norm().item()
                if name not in model.grad_norm_history:
                    model.grad_norm_history[name] = []
                model.grad_norm_history[name].append(gnorm)

        optimizer.step()
        if batch_idx % 100 == 0 and out > 0:
            pred = output.argmax(dim=1, keepdim=True)
            correct = pred.eq(target.view_as(pred)).sum().item()
            print(f'\tTrain Epoch: {epoch:2d} [{batch_idx * len(data):5d}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):2.0f}%)]\tLoss: {loss.item():.6f}\tAccuracy: {correct}/{len(data)}')


def test(model, device, test_loader, out):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    if out > 0:
        print(f'Test set:\tAverage loss: {test_loss:.4f}, Accuracy: {correct:5d}/{len(test_loader.dataset)} ({100. * correct / len(test_loader.dataset):.2f}%)')
    return correct


def plot_grad_norms(model, smooth=True, window=10):
    """Plot the gradient norm evolution for each layer."""
    if not hasattr(model, 'grad_norm_history'):
        print("No gradient history found.")
        return

    plt.figure(figsize=(12, 6))
    for name, norms in model.grad_norm_history.items():
        if smooth and len(norms) > window:
            norms = np.convolve(norms, np.ones(window)/window, mode='valid')
        plt.plot(norms, label=name)

    plt.title("Gradient Norms During Training")
    plt.xlabel("Training Step")
    plt.ylabel("||∇||")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('cifar10_gradients.png')
    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--net_type', type=str, default='H1_J1')
    parser.add_argument('--n_layers', type=int, default=4)
    args = parser.parse_args()

    use_cuda = torch.cuda.is_available()
    batch_size = 128  # Increased batch size for CIFAR-10
    test_batch_size = 1000
    lr = 0.001  # Reduced learning rate for CIFAR-10
    gamma = 0.8
    epochs = 50  # Increased epochs for CIFAR-10
    seed = np.random.randint(0, 1000)
    torch.manual_seed(seed)
    np.random.seed(seed)

    out = 1

    if args.net_type == 'MS1':
        h = 0.4
        wd = 1e-4  # Reduced weight decay for CIFAR-10
        alpha = 1e-4
    elif args.net_type == 'H1_J1':
        h = 0.5
        wd = 1e-4
        alpha = 1e-4
    elif args.net_type == 'H1_J2':
        h = 0.05
        wd = 2e-5
        alpha = 1e-4
    elif args.net_type == 'H2':
        h = 0.5
        wd = 1e-4
        alpha = 1e-4
    else:
        raise ValueError("%s model is not yet implemented" % args.net_type)

    # Define the net model
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'num_workers': 20, 'pin_memory': True} if use_cuda else {}
    model = Net(nf=64, n_layers=args.n_layers, h=h, net_type=args.net_type).to(device)  # Increased nf for CIFAR-10

    print("\n------------------------------------------------------------------")
    print(f"CIFAR-10 dataset - {args.net_type}-DNN - {args.n_layers} layers")
    print(f"== sgd with Adam (lr={lr:.1e}, weight_decay={wd:.1e}, gamma={gamma:.1f}, max_epochs={epochs}, alpha={alpha:.1e}, minibatch={batch_size})")

    best_acc = 0
    best_acc_train = 0

    # Define data transforms with augmentation for CIFAR-10
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    # Load train data
    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('data', train=True, download=True,
                       transform=transform_train),
        batch_size=batch_size, shuffle=True, **kwargs)
    
    # Load test data
    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('data', train=False, transform=transform_test),
        batch_size=test_batch_size, shuffle=True, **kwargs)

    # Define optimization algorithm
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

    # Scheduler for learning_rate parameter
    scheduler = StepLR(optimizer, step_size=1, gamma=gamma)

    # Training
    for epoch in range(1, epochs + 1):
        train(model, device, train_loader, optimizer, epoch, alpha, out)
        test_acc = test(model, device, test_loader, out)
        # Results over training set after training
        train_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in train_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                train_loss += F.nll_loss(output, target, reduction='sum').item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
        train_loss /= len(train_loader.dataset)
        if out > 0:
            print(f'Train set:\tAverage loss: {train_loss:.4f}, Accuracy: {correct:5d}/{len(train_loader.dataset)} ({100. * correct / len(train_loader.dataset):.2f}%)')
        scheduler.step()
        if best_acc < test_acc:
            best_acc = test_acc
            best_acc_train = correct

    plot_grad_norms(model)
    print("\nNetwork trained!")
    print(f'Test accuracy: {100. * best_acc / len(test_loader.dataset):.2f}%  - Train accuracy: {100. * best_acc_train / len(train_loader.dataset):.3f}% ')
    print("------------------------------------------------------------------\n") 