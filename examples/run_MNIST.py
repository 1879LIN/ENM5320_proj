#!/usr/bin/env python
"""
Train a H-DNN on MNIST dataset.
Author: Clara Galimberti (clara.galimberti@epfl.ch)
Usage:
python run_MNIST.py          --net_type      [MODEL NAME]            \
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

from integrators.integrators import MS1, H1, H2, H2_Global, H1_Global
from regularization.regularization import regularization
import argparse
import matplotlib.pyplot as plt
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
            raise ValueError("%s model is not yet implemented for MNIST" % net_type)
        #self.hamiltonian = MS1(n_layers=n_layers, t_end=h * n_layers, nf=nf)
        self.fc_end = nn.Linear(nf, 10)
        self.nf = nf

    def forward(self, x):
        x = self.conv1(x)                        # [B, nf, 28, 28]
        x = x.view(x.size(0), self.nf, -1)       # [B, nf, 784]
        x = x.mean(dim=2, keepdim=True)          # [B, nf, 1]
        x = self.hamiltonian(x)
        x = x.view(x.size(0), -1)                # [B, nf]
        x = self.fc_end(x)
        output = F.log_softmax(x, dim=1)
        return output
    
class Net_Global(nn.Module):
    def __init__(self, nf=8, n_layers=4, h=0.5, net_type='H1_J1'):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1),  # [B, 16, 14, 14]
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),  # [B, 32, 7, 7]
            nn.ReLU(),
            nn.Flatten(),  # [B, 1568]
            nn.Linear(1568, nf)  # nf = p + q
        )

        #self.hamiltonian = H2_Global(n_layers=n_layers, t_end=h * n_layers, nf=nf)
        if net_type == 'H2':
            self.hamiltonian = H2_Global(n_layers=n_layers, t_end=h * n_layers, nf=nf)
        elif net_type == 'H1_J1':
            self.hamiltonian = H1_Global(n_layers=n_layers, t_end=h * n_layers, nf=nf, select_j='J1')
        elif net_type == 'H1_J2':
            self.hamiltonian = H1_Global(n_layers=n_layers, t_end=h * n_layers, nf=nf, select_j='J2')
        else:
            raise ValueError(f"Unsupported net_type: {net_type}")
        self.fc = nn.Linear(nf, 10)

    def forward(self, x):
        x = self.encoder(x)         # [B, nf]
        x = x.unsqueeze(2)          # [B, nf, 1]
        x = self.hamiltonian(x)     # [B, nf, 1]
        x = x.squeeze(2)            # [B, nf]
        x = self.fc(x)              # [B, 10]
        return F.log_softmax(x, dim=1)

    def encode(self, x):
        x = self.encoder(x)     # [B, nf]
        x = x.unsqueeze(2)      # [B, nf, 1]
        return x
    
class CNN_DNN(nn.Module):
    def __init__(self, nf=8, n_layers=2):  # n_layers: number of dense layers after encoder
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),  # [B, 16, 14, 14]
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),  # [B, 32, 7, 7]
            nn.ReLU(),
            nn.Flatten(),       # [B, 1568]
            nn.Linear(1568, nf),
            nn.ReLU()
        )

        # Dynamically add n_layers - 1 linear-ReLU blocks
        dense_blocks = []
        for _ in range(n_layers - 1):  # minus the one already added
            dense_blocks.append(nn.Linear(nf, nf))
            dense_blocks.append(nn.ReLU())

        self.middle_layers = nn.Sequential(*dense_blocks)
        self.fc = nn.Linear(nf, 10)  # final classifier

    def forward(self, x):
        x = self.encoder(x)             # [B, nf]
        x = self.middle_layers(x)      # [B, nf]
        x = self.fc(x)                 # [B, 10]
        return F.log_softmax(x, dim=1)



# def train(model, device, train_loader, optimizer, epoch, alpha, out):
#     model.train()
#     for batch_idx, (data, target) in enumerate(train_loader):
#         data, target = data.to(device), target.to(device)
#         optimizer.zero_grad()
#         output = model(data)
#         loss = F.nll_loss(output, target)
#         K = model.hamiltonian.getK()
#         b = model.hamiltonian.getb()
#         for j in range(int(model.hamiltonian.n_layers) - 1):
#             loss = loss + regularization(alpha, h, K, b)
#         loss.backward()
#         optimizer.step()
#         if batch_idx % 100 == 0 and out>0:
#             output = model(data)
#             pred = output.argmax(dim=1, keepdim=True)
#             correct = pred.eq(target.view_as(pred)).sum().item()
#             print('\tTrain Epoch: {:2d} [{:5d}/{} ({:2.0f}%)]\tLoss: {:.6f}\tAccuracy: {}/{}'.format(
#                 epoch, batch_idx * len(data), len(train_loader.dataset),
#                 100. * batch_idx / len(train_loader), loss.item(), correct, len(data)))

def get_intermediate_states(model, Y0):
    """
    Track intermediate states in the Hamiltonian evolution.
    Y0: [B, nf, 1]
    Returns: [Y0, Y1, ..., Y_T]
    """
    Y0.requires_grad_(True)
    Y_out = [Y0]

    for j in range(model.hamiltonian.n_layers):
        Y = model.hamiltonian(Y_out[j], ini=j, end=j+1)
        Y.retain_grad()
        Y_out.append(Y)

    return Y_out


def train(model, device, train_loader, optimizer, epoch, alpha, out):
    from integrators.CNN_DNN import CNN_DNN
    if isinstance(model, CNN_DNN):
        raise RuntimeError("train() should not be called for CNN_DNN. Use train_cnn_dnn instead.")
    model.train()

    # Initialize gradient history tracker once
    if not hasattr(model, 'grad_norm_history'):
        model.grad_norm_history = [[] for _ in range(model.hamiltonian.n_layers)]

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()

        # Use encode if available (for Net_Global), otherwise fallback
        if hasattr(model, 'encode'):
            latent = model.encode(data)            # [B, nf, 1]
        else:
            latent = model.conv1(data)
            latent = latent.view(latent.size(0), model.nf, -1)
            latent = latent.mean(dim=2, keepdim=True)

        # Get Hamiltonian intermediate states
        Y_out = get_intermediate_states(model, latent)

        # Use the correct final linear layer for each model
        if hasattr(model, 'fc_end'):
            logits = model.fc_end(Y_out[-1].squeeze(2))  # For Net
        else:
            logits = model.fc(Y_out[-1].squeeze(2))      # For Net_Global
        loss = F.nll_loss(F.log_softmax(logits, dim=1), target)

        # Add Hamiltonian parameter regularization
        K = model.hamiltonian.getK()
        b = model.hamiltonian.getb()
        for j in range(model.hamiltonian.n_layers - 1):
            loss = loss + regularization(alpha, model.hamiltonian.h, K, b)

        # Backprop
        loss.backward()

        # === Log gradient norms of Y_j ===
        for j in range(1, len(Y_out)):
            if Y_out[j].grad is not None:
                gnorm = Y_out[j].grad.norm().item()
                model.grad_norm_history[j - 1].append(gnorm)

        # Optional debug print
        if batch_idx % 100 == 0 and out > 0:
            print(f"🌀 Gradient Norms at Epoch {epoch}, Batch {batch_idx}")
            for j in range(model.hamiltonian.n_layers):
                if len(model.grad_norm_history[j]) > 0:
                    print(f"  ‣ Layer {j+1}: ||∇Y_j|| = {model.grad_norm_history[j][-1]:.6f}")

        # Update weights
        optimizer.step()

        # Print training progress
        if batch_idx % 100 == 0 and out > 0:
            pred = logits.argmax(dim=1, keepdim=True)
            correct = pred.eq(target.view_as(pred)).sum().item()
            print('\tTrain Epoch: {:2d} [{:5d}/{} ({:2.0f}%)]\tLoss: {:.6f}\tAccuracy: {}/{}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item(), correct, len(data)))

def train_cnn_dnn(model, device, train_loader, optimizer, epoch, out):
    model.train()

    # Initialize gradient history tracker
    if not hasattr(model, 'grad_norm_history'):
        model.grad_norm_history = {}

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()

        # Forward pass
        output = model(data)  # [B, 10]
        loss = F.nll_loss(output, target)

        # Backward pass
        loss.backward()

        # Track gradients of fully connected layers
        for name, param in model.named_parameters():
            if param.grad is not None and 'fc' in name or 'middle_layers' in name:
                if name not in model.grad_norm_history:
                    model.grad_norm_history[name] = []
                model.grad_norm_history[name].append(param.grad.norm().item())

        # Optional debug print
        if batch_idx % 100 == 0 and out > 0:
            print(f"🧠 CNN-DNN Gradient Norms at Epoch {epoch}, Batch {batch_idx}")
            for name, norms in model.grad_norm_history.items():
                if len(norms) > 0:
                    print(f"  ‣ {name}: ||∇|| = {norms[-1]:.6f}")

        optimizer.step()

        # Print training progress
        if batch_idx % 100 == 0 and out > 0:
            pred = output.argmax(dim=1, keepdim=True)
            correct = pred.eq(target.view_as(pred)).sum().item()
            print('\tTrain Epoch: {:2d} [{:5d}/{} ({:2.0f}%)]\tLoss: {:.6f}\tAccuracy: {}/{}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item(), correct, len(data)))


def test(model, device, test_loader, out):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    if out > 0:
        print('Test set:\tAverage loss: {:.4f}, Accuracy: {:5d}/{} ({:.2f}%)'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))
    return correct

def plot_grad_norms(model, smooth=True, window=10, save_path=None, title=None, ylim=None, label=None):
    """
    Improved plot of gradient norm evolution for each layer.
    Handles both list and dict grad_norm_history.
    """
    if not hasattr(model, 'grad_norm_history'):
        print("No gradient history found.")
        return

    import matplotlib.pyplot as plt
    import numpy as np

    plt.figure(figsize=(10, 5))
    font = {'size': 14}
    plt.rc('font', **font)

    grad_hist = model.grad_norm_history
    if isinstance(grad_hist, dict):
        items = grad_hist.items()
    else:
        items = [(f'Layer {i+1}', gnorms) for i, gnorms in enumerate(grad_hist)]

    for name, gnorms in items:
        if smooth and len(gnorms) > window:
            gnorms = np.convolve(gnorms, np.ones(window)/window, mode='valid')
        plt.plot(gnorms, label=str(name))

    if title:
        plt.title(title, fontsize=16)
    else:
        plt.title("Gradient Norms per Layer During Training", fontsize=16)
    plt.xlabel("Training Step", fontsize=14)
    plt.ylabel(r"$||\nabla Y_j||$", fontsize=14)
    plt.legend(fontsize=12, ncol=2)
    plt.grid(True, linestyle='--', alpha=0.7)
    if ylim:
        plt.ylim(*ylim)
    if label:
        plt.text(-0.1, 1.05, label, transform=plt.gca().transAxes, fontsize=16, fontweight='bold', va='top')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--net_type', type=str, default='H1_J1')
    parser.add_argument('--n_layers', type=int, default=1)
    args = parser.parse_args()

    use_cuda = torch.cuda.is_available()  # not no_cuda and
    batch_size = 100
    test_batch_size = 1000
    lr = 0.01
    gamma = 0.8
    # epochs = 10
    epochs = 1
    seed = np.random.randint(0, 1000)
    torch.manual_seed(seed)
    np.random.seed(seed)

    out = 1

    if args.net_type == 'MS1':
        h = 0.4
        wd = 1e-3
        alpha = 1e-3
    elif args.net_type == 'H1_J1':
        h = 0.5
        wd = 1e-3
        alpha = 1e-3
    elif args.net_type == 'H1_J2':
        h = 0.05
        wd = 2e-4
        alpha = 1e-3
    elif args.net_type == 'H2':
        h = 0.5
        wd = 1e-3
        alpha = 1e-3
    elif args.net_type == 'CNN_DNN':
        h = None  # Not used for CNN_DNN
        wd = 1e-3
        alpha = 1e-3
    else:
        raise ValueError("%s model is not yet implemented" % args.net_type)

    # Set nf based on model type
    if args.net_type == 'CNN_DNN':
        nf = 32
    else:
        nf = 16

    # Define the net model
    print("Instantiating Net with nf =", nf)
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'num_workers': 20, 'pin_memory': True} if use_cuda else {}
    if args.net_type == 'CNN_DNN':
        model = CNN_DNN(nf=nf, n_layers=args.n_layers).to(device)
    else:
        model = Net_Global(nf=nf, n_layers=args.n_layers, h=h, net_type=args.net_type).to(device)


    print("\n------------------------------------------------------------------")
    print("MNIST dataset - %s-DNN - %i layers" % (args.net_type, args.n_layers))
    print("== sgd with Adam (lr=%.1e, weight_decay=%.1e, gamma=%.1f, max_epochs=%i, alpha=%.1e, minibatch=%i)" %
          (lr, wd, gamma, epochs, alpha, batch_size))

    best_acc = 0
    best_acc_train = 0

    # Load train data
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=batch_size, shuffle=True, **kwargs)
    # Load test data
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('data', train=False, transform=transforms.Compose([
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
                train_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()
        train_loss /= len(train_loader.dataset)
        if out > 0:
            print('Train set:\tAverage loss: {:.4f}, Accuracy: {:5d}/{} ({:.2f}%)'.format(
                train_loss, correct, len(train_loader.dataset),
                100. * correct / len(train_loader.dataset)))
        scheduler.step()
        if best_acc < test_acc:
            best_acc = test_acc
            best_acc_train = correct

    print("\nNetwork trained!")
    print('Test accuracy: {:.2f}%  - Train accuracy: {:.3f}% '.format(
         100. * best_acc / len(test_loader.dataset), 100. * best_acc_train / len(train_loader.dataset)))
    
    if args.net_type != 'CNN_DNN':
        plot_grad_norms(
            model,
            save_path=f"mnist_grad_norms_{args.n_layers}_layers.png",
            title=f"Gradient Norms During Training: HDNN ({args.n_layers} Layers)",
            ylim=(0, 0.05)
            #label=f"({args.n_layers})"
        )
    
    print("------------------------------------------------------------------\n")