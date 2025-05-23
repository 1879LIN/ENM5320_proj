#!/usr/bin/env python
"""
Train a H-DNN on Fashion MNIST dataset.
Author: Clara Galimberti (clara.galimberti@epfl.ch)
Usage:
python run_FashionMNIST.py    --net_type      [MODEL NAME]            \
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
from integrators.CNN_DNN import CNN_DNN, train_cnn_dnn


class Net(nn.Module):
    def __init__(self, nf=8, n_layers=4, h=0.5, net_type='H1_J1'):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=nf, kernel_size=3, stride=1, padding=1)
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
        
        self.fc_end = nn.Linear(nf*28*28, 10)
        self.nf = nf

    def forward(self, x):
        x = self.conv1(x)
        x = x.view(x.size(0), self.nf, -1)
        x = self.hamiltonian(x)
        x = x.reshape(-1, self.nf*28*28)
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

    # Get number of layers based on model type
    if hasattr(model, 'hamiltonian'):
        n_layers = model.hamiltonian.n_layers
    elif hasattr(model, 'n_layers'):
        n_layers = model.n_layers
    else:
        n_layers = sum(1 for m in model.modules() if isinstance(m, nn.Conv2d))

    plt.title(f"Gradient Norms During Training - {n_layers} Layers")
    plt.xlabel("Training Step")
    plt.ylabel("||∇||")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'fashion_mnist_gradients_{n_layers}layers.png')
    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--net_type', type=str, default='H1_J1')
    parser.add_argument('--n_layers', type=int, default=4)
    args = parser.parse_args()

    use_cuda = torch.cuda.is_available()
    batch_size = 100
    test_batch_size = 1000
    lr = 0.001
    gamma = 0.8
    epochs = 3  # Increased epochs for Fashion MNIST
    seed = np.random.randint(0, 1000)
    torch.manual_seed(seed)
    np.random.seed(seed)

    out = 1

    if args.net_type == 'MS1':
        h = 0.4
        wd = 1.5e-3  # Increased weight decay
        alpha = 8e-4  # Adjusted regularization
    elif args.net_type == 'H1_J1':
        h = 0.5
        wd = 1.5e-3  # Increased weight decay
        alpha = 8e-4  # Adjusted regularization
    elif args.net_type == 'H1_J2':
        h = 0.05
        wd = 3e-4  # Increased weight decay
        alpha = 1e-3
    elif args.net_type == 'H2':
        h = 0.5
        wd = 1.5e-3  # Increased weight decay
        alpha = 8e-4  # Adjusted regularization
    elif args.net_type == 'CNN_DNN':
        h = None  # Not used for CNN_DNN
        wd = 1e-3  # Keep original weight decay for CNN
        alpha = 1e-3  # Keep original regularization for CNN
    else:
        raise ValueError("%s model is not yet implemented" % args.net_type)

    # Define the net model
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'num_workers': 20, 'pin_memory': True} if use_cuda else {}
    if args.net_type == 'CNN_DNN':
        model = CNN_DNN(nf=32, n_layers=args.n_layers).to(device)  # Keep original features for CNN
    else:
        model = Net(nf=48, n_layers=args.n_layers, h=h, net_type=args.net_type).to(device)  # Increased features for HNN

    print("\n------------------------------------------------------------------")
    print(f"Fashion MNIST dataset - {args.net_type}-DNN - {args.n_layers} layers")
    print(f"== sgd with Adam (lr={lr:.1e}, weight_decay={wd:.1e}, gamma={gamma:.1f}, max_epochs={epochs}, alpha={alpha:.1e}, minibatch={batch_size})")

    best_acc = 0
    best_acc_train = 0

    # Load train data
    train_loader = torch.utils.data.DataLoader(
        datasets.FashionMNIST('data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=batch_size, shuffle=True, **kwargs)
    
    # Load test data
    test_loader = torch.utils.data.DataLoader(
        datasets.FashionMNIST('data', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=test_batch_size, shuffle=True, **kwargs)

    # Define optimization algorithm
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

    # Scheduler for learning_rate parameter
    scheduler = StepLR(optimizer, step_size=1, gamma=gamma)

    # Training
    for epoch in range(1, epochs + 1):
        if args.net_type == 'CNN_DNN':
            train_cnn_dnn(model, device, train_loader, optimizer, epoch, out)
        else:
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
