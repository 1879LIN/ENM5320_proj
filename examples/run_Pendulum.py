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
if not hasattr(np, 'bool8'):
    np.bool8 = np.bool_
import argparse
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
import gym
import os

# Import your models here
from integrators.integrators import H1, H2, MS1
from integrators.CNN_DNN import CNN_DNN, train_cnn_dnn
from regularization.regularization import regularization
# from your_project import Net_Global  # If you want to use Net_Global

class FeedforwardNet(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=64, output_dim=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, output_dim)
        )
    def forward(self, x):
        return self.net(x)

class SimpleHNN(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh()
        )
        self.hamiltonian = nn.Linear(hidden_dim, 1)  # Scalar Hamiltonian
        self.input_dim = input_dim

    def forward(self, x):
        x = x.clone().detach().requires_grad_(True)
        z = self.encoder(x)
        H = self.hamiltonian(z)
        grad = torch.autograd.grad(H.sum(), x, create_graph=True)[0]
        return x + grad * 0.05  # Euler step

class StrictHNN(nn.Module):
    def __init__(self, nf=16, n_layers=4, h=0.05, net_type='H1_J1'):
        super().__init__()
        # 更深更宽的编码器，三层Tanh
        self.encoder_q = nn.Sequential(
            nn.Linear(2, 32),
            nn.Tanh(),
            nn.Linear(32, 16),
            nn.Tanh(),
            nn.Linear(16, nf//2),
            nn.Tanh()
        )
        self.encoder_p = nn.Sequential(
            nn.Linear(1, 32),
            nn.Tanh(),
            nn.Linear(32, 16),
            nn.Tanh(),
            nn.Linear(16, nf//2),
            nn.Tanh()
        )
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
        # 更深的解码器，三层Tanh
        self.decoder = nn.Sequential(
            nn.Linear(nf, 32),
            nn.Tanh(),
            nn.Linear(32, 16),
            nn.Tanh(),
            nn.Linear(16, 3)
        )
        self.nf = nf

    def forward(self, x):
        q = self.encoder_q(x[:, :2])
        p = self.encoder_p(x[:, 2:])
        y = torch.cat([q, p], dim=1).unsqueeze(-1)
        y = self.hamiltonian(y)
        y = y.squeeze(-1)
        out = self.decoder(y)
        return out

def generate_pendulum_data(n_samples=5000, t_max=1.0, dt=0.05, g=9.81, L=1.0, noise_std=0.05):
    """
    Generate synthetic pendulum data by integrating the equations of motion.
    Returns:
        X: [n_samples, 3] (cosθ, sinθ, ω) at t=0
        y: [n_samples, 1] θ at t=t_max
    """
    X = []
    y = []
    for _ in range(n_samples):
        # Random initial angle and angular velocity
        theta0 = np.random.uniform(-np.pi, np.pi)
        omega0 = np.random.uniform(-2, 2)
        theta, omega = theta0, omega0

        # Integrate forward in time
        n_steps = int(t_max / dt)
        for _ in range(n_steps):
            dtheta = omega
            domega = - (g / L) * np.sin(theta)
            theta += dtheta * dt
            omega += domega * dt

        # Add noise to input
        state = np.array([np.cos(theta0), np.sin(theta0), omega0]) + np.random.normal(0, noise_std, 3)
        X.append(state)
        y.append(theta)  # Target is the angle at t = t_max

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32).reshape(-1, 1)
    return X, y

def load_pendulum_data(batch_size, test_batch_size, n_train=5000, n_test=1000):
    X_train, y_train = generate_pendulum_data(n_samples=n_train)
    X_test, y_test = generate_pendulum_data(n_samples=n_test)
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=test_batch_size, shuffle=False)
    print(f"Training data shape: {X_train.shape}, labels shape: {y_train.shape}")
    print(f"Test data shape: {X_test.shape}, labels shape: {y_test.shape}")
    return train_loader, test_loader

def generate_pendulum_trajectories(n_trajectories=1000, trajectory_length=50, dt=0.05):
    env = gym.make('Pendulum-v1')
    X = []
    Y = []
    for _ in range(n_trajectories):
        obs = env.reset()
        if isinstance(obs, tuple):  # Gym >=0.26
            obs = obs[0]
        for _ in range(trajectory_length):
            action = np.array([0.0])  # No torque
            next_obs, _, done, _, _ = env.step(action)
            X.append(obs)
            Y.append(next_obs)
            obs = next_obs
    X = np.array(X, dtype=np.float32)
    Y = np.array(Y, dtype=np.float32)
    return X, Y

def load_time_evolution_data(batch_size, test_batch_size, n_train=40000, n_test=10000):
    X, Y = generate_pendulum_trajectories(n_trajectories=(n_train+n_test)//50, trajectory_length=50)
    X_train, Y_train = X[:n_train], Y[:n_train]
    X_test, Y_test = X[n_train:n_train+n_test], Y[n_train:n_train+n_test]
    train_loader = DataLoader(TensorDataset(torch.tensor(X_train), torch.tensor(Y_train)), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(TensorDataset(torch.tensor(X_test), torch.tensor(Y_test)), batch_size=test_batch_size, shuffle=False)
    print(f"Time-evolution Training data shape: {X_train.shape}, labels shape: {Y_train.shape}")
    print(f"Time-evolution Test data shape: {X_test.shape}, labels shape: {Y_test.shape}")
    return train_loader, test_loader

# Adjust Net to accept vector input [B, 3]
class Net(nn.Module):
    def __init__(self, nf=8, n_layers=4, h=0.5, net_type='H1_J1'):
        super().__init__()
        # Input encoder to map 3D state to nf-dimensional space
        self.encoder = nn.Sequential(
            nn.Linear(3, nf//2),  # Map to half the features (position coordinates)
            nn.Tanh()
        )
        
        # Initialize momentum coordinates as zeros
        self.register_buffer('p0', torch.zeros(1, nf//2, 1))
        
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
            
        # Output decoder to map back to angle prediction
        self.decoder = nn.Sequential(
            nn.Linear(nf//2, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        self.nf = nf
        
    def forward(self, x):
        # Encode position coordinates
        q = self.encoder(x)  # [B, nf/2]
        q = q.unsqueeze(-1)  # [B, nf/2, 1]
        
        # Initialize momentum coordinates
        p = self.p0.expand(x.size(0), -1, -1)  # [B, nf/2, 1]
        
        # Concatenate position and momentum
        y = torch.cat([q, p], dim=1)  # [B, nf, 1]
        
        # Pass through Hamiltonian layers
        y = self.hamiltonian(y)
        
        # Extract position coordinates and decode to angle
        q_out = y[:, :self.nf//2, 0]  # [B, nf/2]
        angle = self.decoder(q_out)  # [B, 1]
        
        return angle

class PendulumNet(nn.Module):
    def __init__(self, nf=32, n_layers=4):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(3, nf),
            nn.ReLU(),
            nn.Dropout(0.2)  # Add dropout for regularization
        )
        
        dense_blocks = []
        for _ in range(n_layers - 1):
            dense_blocks.append(nn.Linear(nf, nf))
            dense_blocks.append(nn.ReLU())
            dense_blocks.append(nn.Dropout(0.2))
            
        self.middle_layers = nn.Sequential(*dense_blocks)
        self.fc = nn.Linear(nf, 1)  # Output single value for angle prediction
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.middle_layers(x)
        x = self.fc(x)
        return x

def train_pendulum(model, device, train_loader, optimizer, epoch, out):
    model.train()
    total_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.mse_loss(output, target)  # Use MSE loss for regression
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        
        if batch_idx % 100 == 0 and out > 0:
            print(f'\tTrain Epoch: {epoch:2d} [{batch_idx * len(data):5d}/({len(train_loader.dataset)})] Loss: {loss.item():.6f}')

def train(model, device, train_loader, optimizer, epoch, alpha, out):
    model.train()
    total_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.mse_loss(output, target)  # Use MSE loss for regression
        K = model.hamiltonian.getK()
        b = model.hamiltonian.getb()
        for j in range(int(model.hamiltonian.n_layers) - 1):
            loss = loss + regularization(alpha, model.hamiltonian.h, K, b)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        
        if batch_idx % 100 == 0 and out > 0:
            print(f'\tTrain Epoch: {epoch:2d} [{batch_idx * len(data):5d}/({len(train_loader.dataset)})] Loss: {loss.item():.6f}')

def test(model, device, test_loader, out):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.mse_loss(output, target, reduction='sum').item()
    test_loss /= len(test_loader.dataset)
    if out > 0:
        print(f'Test set: Average loss: {test_loss:.4f}, RMSE: {np.sqrt(test_loss):.4f} radians')
    return test_loss

def train_time_evolution(model, device, train_loader, optimizer, epoch, out):
    model.train()
    total_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.mse_loss(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        if batch_idx % 100 == 0 and out > 0:
            print(f'\tTrain Epoch: {epoch:2d} [{batch_idx * len(data):5d}/({len(train_loader.dataset)})] Loss: {loss.item():.6f}')

def test_time_evolution(model, device, test_loader, out):
    model.eval()
    test_loss = 0
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        test_loss += F.mse_loss(output, target, reduction='sum').item()
    test_loss /= len(test_loader.dataset)
    if out > 0:
        print(f'Test set: Average loss: {test_loss:.4f}, RMSE: {np.sqrt(test_loss):.4f}')
    return test_loss

def generate_generalization_trajectories(n_train=40000, n_test=10000, trajectory_length=10, dt=0.05):
    env = gym.make('Pendulum-v1')
    X_train, Y_train = [], []
    X_test, Y_test = [], []
    # 训练集：初始角度在[-pi/2, pi/2]
    for _ in range(n_train):
        theta0 = np.random.uniform(-np.pi/2, np.pi/2)
        omega0 = np.random.uniform(-2, 2)
        obs = np.array([np.cos(theta0), np.sin(theta0), omega0], dtype=np.float32)
        states = [obs.copy()]
        theta, omega = theta0, omega0
        for _ in range(trajectory_length):
            dtheta = omega
            domega = -9.81 * np.sin(theta)
            theta += dtheta * dt
            omega += domega * dt
            obs = np.array([np.cos(theta), np.sin(theta), omega], dtype=np.float32)
            states.append(obs.copy())
        X_train.append(states[0])
        Y_train.append(np.stack(states[1:]))  # shape: [n_steps, 3]
    # 测试集：初始角度在[-pi, -pi/2]和[pi/2, pi]
    for _ in range(n_test):
        if np.random.rand() < 0.5:
            theta0 = np.random.uniform(-np.pi, -np.pi/2)
        else:
            theta0 = np.random.uniform(np.pi/2, np.pi)
        omega0 = np.random.uniform(-2, 2)
        obs = np.array([np.cos(theta0), np.sin(theta0), omega0], dtype=np.float32)
        states = [obs.copy()]
        theta, omega = theta0, omega0
        for _ in range(trajectory_length):
            dtheta = omega
            domega = -9.81 * np.sin(theta)
            theta += dtheta * dt
            omega += domega * dt
            obs = np.array([np.cos(theta), np.sin(theta), omega], dtype=np.float32)
            states.append(obs.copy())
        X_test.append(states[0])
        Y_test.append(np.stack(states[1:]))
    X_train = np.array(X_train, dtype=np.float32)
    Y_train = np.array(Y_train, dtype=np.float32)  # [N, n_steps, 3]
    X_test = np.array(X_test, dtype=np.float32)
    Y_test = np.array(Y_test, dtype=np.float32)
    return X_train, Y_train, X_test, Y_test

def load_generalization_data(batch_size, test_batch_size, n_train=1000, n_test=20000, n_steps=50):
    X_train, Y_train, X_test, Y_test = generate_generalization_trajectories(n_train, n_test, n_steps)
    train_loader = DataLoader(TensorDataset(torch.tensor(X_train), torch.tensor(Y_train)), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(TensorDataset(torch.tensor(X_test), torch.tensor(Y_test)), batch_size=test_batch_size, shuffle=False)
    print(f"Generalization Training data shape: {X_train.shape}, labels shape: {Y_train.shape}, n_train={n_train}, n_steps={n_steps}")
    print(f"Generalization Test data shape: {X_test.shape}, labels shape: {Y_test.shape}, n_test={n_test}, n_steps={n_steps}")
    return train_loader, test_loader

# 能量守恒正则项
# H = 0.5 * p^2 - cos(theta)
def energy_regularization(x, y, nf):
    # x: [B, 3] (cosθ, sinθ, ω) 输入
    # y: [B, nf] (q, p) after Hamiltonian block
    # 只对第一个 q 分量和 p 分量做能量约束
    # q1 = y[:, 0], p1 = y[:, nf//2]
    q1 = y[:, 0]
    p1 = y[:, nf//2]
    # 由 sinθ, cosθ 还原 θ
    theta = torch.atan2(x[:, 1], x[:, 0])
    H_true = 0.5 * x[:, 2] ** 2 - torch.cos(theta)
    H_pred = 0.5 * p1 ** 2 - torch.cos(q1)
    return F.mse_loss(H_pred, H_true)

def train_multistep(model, device, train_loader, optimizer, epoch, n_steps, out, alpha=1e-4):
    model.train()
    for batch_idx, (data, targets) in enumerate(train_loader):
        data, targets = data.to(device), targets.to(device)
        optimizer.zero_grad()
        x = data
        loss = 0
        energy_reg = 0
        for t in range(n_steps):
            if isinstance(model, StrictHNN):
                q = model.encoder_q(x[:, :2])
                p = model.encoder_p(x[:, 2:])
                y_ham = torch.cat([q, p], dim=1).unsqueeze(-1)
                y_ham = model.hamiltonian(y_ham).squeeze(-1)
                pred = model.decoder(y_ham)
                loss = loss + F.mse_loss(pred, targets[:, t, :])
                energy_reg = energy_reg + energy_regularization(x, y_ham, model.nf)
                x = pred
            else:
                x = model(x)
                loss = loss + F.mse_loss(x, targets[:, t, :])
        loss = (loss + alpha * energy_reg) / n_steps
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0 and out > 0:
            print(f'\tTrain Epoch: {epoch:2d} [{batch_idx * len(data):5d}/({len(train_loader.dataset)})] Loss: {loss.item():.6f}')

def test_multistep(model, device, test_loader, n_steps, out):
    model.eval()
    test_loss = 0
    for data, targets in test_loader:
        data, targets = data.to(device), targets.to(device)
        x = data
        for t in range(n_steps):
            if isinstance(model, SimpleHNN):
                x = x.clone().detach().requires_grad_(True)
            x = model(x)
            test_loss += F.mse_loss(x, targets[:, t, :], reduction='sum').item()
    test_loss /= (len(test_loader.dataset) * n_steps)
    if out > 0:
        print(f'Test set: Average multi-step loss: {test_loss:.4f}, RMSE: {np.sqrt(test_loss):.4f}')
    return test_loss

def plot_multistep_trajectory(model, device, test_loader, n_steps, n_traj=3, title=''):
    model.eval()
    data_iter = iter(test_loader)
    data, targets = next(data_iter)
    data, targets = data.to(device), targets.to(device)
    idxs = np.random.choice(len(data), n_traj, replace=False)
    x0s = data[idxs]
    true_trajs = targets[idxs]  # [n_traj, n_steps, 3]

    pred_trajs = []
    x = x0s
    for t in range(n_steps):
        if isinstance(model, StrictHNN):
            q = model.encoder_q(x[:, :2])
            p = model.encoder_p(x[:, 2:])
            y_ham = torch.cat([q, p], dim=1).unsqueeze(-1)
            y_ham = model.hamiltonian(y_ham).squeeze(-1)
            x = model.decoder(y_ham)
        else:
            x = model(x)
        pred_trajs.append(x.detach().cpu().numpy())
    pred_trajs = np.stack(pred_trajs, axis=1)  # [n_traj, n_steps, 3]

    os.makedirs('trajectory_plots', exist_ok=True)
    for i in range(n_traj):
        plt.figure(figsize=(12, 4))
        # 角度轨迹
        true_theta = np.arctan2(true_trajs[i, :, 1].cpu(), true_trajs[i, :, 0].cpu())
        pred_theta = np.arctan2(pred_trajs[i, :, 1], pred_trajs[i, :, 0])
        plt.subplot(1, 2, 1)
        plt.plot(range(n_steps), true_theta, label='True θ')
        plt.plot(range(n_steps), pred_theta, '--', label='Pred θ')
        plt.xlabel('Step')
        plt.ylabel('Angle (rad)')
        plt.title(f'Trajectory {i+1} Angle')
        plt.legend()

        # 角速度轨迹
        plt.subplot(1, 2, 2)
        plt.plot(range(n_steps), true_trajs[i, :, 2].cpu(), label='True ω')
        plt.plot(range(n_steps), pred_trajs[i, :, 2], '--', label='Pred ω')
        plt.xlabel('Step')
        plt.ylabel('Angular velocity')
        plt.title(f'Trajectory {i+1} Angular Velocity')
        plt.legend()

        plt.suptitle(title + f' (Sample {i+1})')
        plt.tight_layout()
        save_path = f'trajectory_plots/trajectory_{title}_sample{i+1}.png'
        plt.savefig(save_path, dpi=150)
        print(f"Saved: {save_path}")
        plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--net_type', type=str, default='H1_J1', choices=['H1_J1', 'H1_J2', 'H2', 'MS1', 'CNN_DNN', 'StrictHNN'])
    parser.add_argument('--n_layers', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--test_batch_size', type=int, default=200)
    parser.add_argument('--task', type=str, default='regression', choices=['regression', 'time_evolution', 'generalization'])
    parser.add_argument('--n_steps', type=int, default=50)
    parser.add_argument('--n_train', type=int, default=1000)
    parser.add_argument('--n_test', type=int, default=20000)
    args = parser.parse_args()

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    nf = 16 if args.net_type == 'StrictHNN' else 32
    h = 0.05 if args.net_type == 'StrictHNN' else 0.5
    out = 1

    if args.task == 'generalization':
        train_loader, test_loader = load_generalization_data(args.batch_size, args.test_batch_size, n_train=args.n_train, n_test=args.n_test, n_steps=args.n_steps)
        if args.net_type == 'CNN_DNN':
            model = FeedforwardNet(input_dim=3, hidden_dim=64, output_dim=3).to(device)
        elif args.net_type == 'StrictHNN':
            model = StrictHNN(nf=nf, n_layers=args.n_layers, h=h, net_type='H1_J1').to(device)
        else:
            model = SimpleHNN(input_dim=3, hidden_dim=64).to(device)
        train_fn = lambda *a, **kw: train_multistep(*a, **kw, alpha=1e-4)
        test_fn = test_multistep
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
        scheduler = StepLR(optimizer, step_size=5, gamma=0.5)
        for epoch in range(1, args.epochs + 1):
            train_fn(model, device, train_loader, optimizer, epoch, args.n_steps, out)
            test_fn(model, device, test_loader, args.n_steps, out)
            scheduler.step()
        print("\nGeneralization (multi-step) training complete.")
        plot_multistep_trajectory(model, device, test_loader, n_steps=args.n_steps, n_traj=3, title=args.net_type)
        return

    if args.task == 'time_evolution':
        train_loader, test_loader = load_time_evolution_data(args.batch_size, args.test_batch_size)
        if args.net_type == 'CNN_DNN':
            model = FeedforwardNet(input_dim=3, hidden_dim=64, output_dim=3).to(device)
        else:
            model = SimpleHNN(input_dim=3, hidden_dim=64).to(device)
        train_fn = train_time_evolution
        test_fn = test_time_evolution
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
        scheduler = StepLR(optimizer, step_size=5, gamma=0.5)
        for epoch in range(1, args.epochs + 1):
            train_fn(model, device, train_loader, optimizer, epoch, out)
            test_fn(model, device, test_loader, out)
            scheduler.step()
        print("\nTime-evolution training complete.")
        return

    # Load Pendulum data
    train_loader, test_loader = load_pendulum_data(args.batch_size, args.test_batch_size)

    # Model selection
    if args.net_type == 'CNN_DNN':
        model = PendulumNet(nf=nf, n_layers=args.n_layers).to(device)
        train_fn = train_pendulum
    else:
        model = Net(nf=nf, n_layers=args.n_layers, h=h, net_type=args.net_type).to(device)
        train_fn = train

    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)  # Adjusted learning rate and weight decay
    scheduler = StepLR(optimizer, step_size=5, gamma=0.5)  # Adjusted scheduler

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