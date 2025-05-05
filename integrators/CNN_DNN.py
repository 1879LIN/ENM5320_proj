import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import numpy as np
import matplotlib.pyplot as plt


class CNN_DNN(nn.Module):
    def __init__(self, nf=8, n_layers=2):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        self.input_fc = nn.Sequential(
            nn.Linear(1568, nf),
            nn.ReLU()
        )

        dense_blocks = []
        for i in range(n_layers - 1):
            dense_blocks.append(nn.Linear(nf, nf))
            dense_blocks.append(nn.ReLU())

        self.middle_layers = nn.Sequential(*dense_blocks)
        self.fc = nn.Linear(nf, 10)
        self.dnn_activations = []

    def forward(self, x):
        x = self.encoder(x)
        self.dnn_activations = []

        for idx, layer in enumerate(self.input_fc):
            x = layer(x)
            if isinstance(layer, nn.ReLU):
                #x.retain_grad()
                if x.requires_grad:
                    x.retain_grad()
                self.dnn_activations.append((f"input_fc_{idx}", x))

        for idx, layer in enumerate(self.middle_layers):
            x = layer(x)
            if isinstance(layer, nn.ReLU):
                #x.retain_grad()
                if x.requires_grad:
                    x.retain_grad()
                self.dnn_activations.append((f"middle_fc_{idx}", x))

        x = self.fc(x)
        return F.log_softmax(x, dim=1)


def train_cnn_dnn(model, device, train_loader, optimizer, epoch, out):
    model.train()
    if not hasattr(model, 'grad_norm_history'):
        model.grad_norm_history = {}

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()

        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()

        for name, act in model.dnn_activations:
            if act.grad is not None:
                if name not in model.grad_norm_history:
                    model.grad_norm_history[name] = []
                model.grad_norm_history[name].append(act.grad.norm().item())

        optimizer.step()

        if batch_idx % 100 == 0 and out > 0:
            print(f"\n🔍 Epoch {epoch}, Batch {batch_idx} — DNN ∥∇Y_j∥:")
            for name, norms in model.grad_norm_history.items():
                if len(norms) > 0:
                    print(f"  ‣ {name}: ||∇|| = {norms[-1]:.6f}")

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
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    if out > 0:
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))
    return correct


def plot_grad_norms(model, smooth=True, window=10):
    if not hasattr(model, 'grad_norm_history'):
        print("No gradient history found.")
        return

    plt.figure(figsize=(12, 6))
    for name, gnorms in model.grad_norm_history.items():
        if smooth and len(gnorms) > window:
            gnorms = np.convolve(gnorms, np.ones(window)/window, mode='valid')
        plt.plot(gnorms, label=name)

    plt.title("Gradient Norms in CNN-DNN Layers")
    plt.xlabel("Training Step")
    plt.ylabel("||∇Y_j||")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    batch_size = 100
    test_batch_size = 1000
    lr = 0.01
    gamma = 0.8
    epochs = 5
    wd = 2e-4
    nf = 16
    n_layers = 5

    out = 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    kwargs = {'num_workers': 2, 'pin_memory': True} if torch.cuda.is_available() else {}

    torch.manual_seed(42)
    np.random.seed(42)

    model = CNN_DNN(nf=nf, n_layers=n_layers).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = StepLR(optimizer, step_size=1, gamma=gamma)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('data', train=True, download=True, transform=transform),
        batch_size=batch_size, shuffle=True, **kwargs)

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('data', train=False, transform=transform),
        batch_size=test_batch_size, shuffle=False, **kwargs)

    for epoch in range(1, epochs + 1):
        train_cnn_dnn(model, device, train_loader, optimizer, epoch, out)
        test(model, device, test_loader, out)
        scheduler.step()

    print("\nTraining Complete.")
    plot_grad_norms(model)
