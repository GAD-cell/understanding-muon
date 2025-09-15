import os
import glob
import pickle
import argparse
import itertools
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from muon import MuonClip, MuonConfig


class SmallNet(nn.Module):
    """A simple 3-layer linear + ReLU network."""
    def __init__(self):
        super().__init__()
        torch.manual_seed(10)
        self.linear = nn.Linear(3, 3, bias=False)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.linear(self.relu(self.linear(x))))


coeffs = {
    "p1": [1.0, 0.5, -0.3, 0.2, 0.1, -0.2, 0.3, -0.1, 0.0, 0.0],
    "p2": [0.5, -0.2, 0.4, 0.1, -0.3, 0.2, -0.1, 0.2, 0.0, 0.1],
    "p3": [-0.3, 0.4, 0.1, -0.2, 0.3, -0.1, 0.2, -0.2, 0.0, -0.1]
}


def poly_eval(x, coefs):
    """Evaluate a quadratic polynomial in 3 variables."""
    x1, x2, x3 = x[:, 0], x[:, 1], x[:, 2]
    a, b, c, d, e, f, g, h, i, j = coefs
    return (
        a * x1**2 + b * x2**2 + c * x3**2
        + d * x1 * x2 + e * x1 * x3 + f * x2 * x3
        + g * x1 + h * x2 + i * x3 + j
    )


def system_poly(X):
    f1 = poly_eval(X, coeffs["p1"])
    f2 = poly_eval(X, coeffs["p2"])
    f3 = poly_eval(X, coeffs["p3"])
    return torch.stack([f1, f2, f3], dim=1)


def train_models(steps=5000):
    """Train Adam and Muon optimizers on the polynomial system."""
    torch.manual_seed(45)
    X = torch.randn(100, 3)
    y = system_poly(X)

    net_adam, net_muon = SmallNet(), SmallNet()
    optim_adam = optim.Adam(net_adam.parameters(), lr=5e-4)
    muon_config = MuonConfig(lr=5e-4, cans_ortho=False, ns_steps=5, enable_clipping=False, log_dir="")
    optim_muon = MuonClip(net_muon, {}, muon_config)

    basis = torch.eye(3)
    traj_adam, traj_muon = [], []
    cumulative_adam_delta_angle, cumulative_muon_delta_angle = [0.0], [0.0]

    for step in range(steps):
        # Adam update
        old_adam_direction = net_adam.linear.weight.clone().T
        optim_adam.zero_grad()
        loss_adam = ((net_adam(X) - y) ** 2).mean()
        loss_adam.backward()
        optim_adam.step()

        with torch.no_grad():
            V = (net_adam.linear.weight @ basis).T
            traj_adam.append(V.numpy())
            for i in range(3):
                vi, vj = V[:, i], old_adam_direction[:, i]
                cos_sim = torch.dot(vi, vj) / (vi.norm() * vj.norm() + 1e-12)
                theta = torch.acos(torch.clamp(cos_sim, -1.0, 1.0)) * 180.0 / torch.pi
                cumulative_adam_delta_angle.append(cumulative_adam_delta_angle[-1] + theta.item())

        # Muon update
        old_muon_direction = net_muon.linear.weight.clone().T
        optim_muon.zero_grad()
        loss_muon = ((net_muon(X) - y) ** 2).mean()
        loss_muon.backward()
        optim_muon.step()

        with torch.no_grad():
            V = (net_muon.linear.weight @ basis).T
            traj_muon.append(V.numpy())
            for i in range(3):
                vi, vj = V[:, i], old_muon_direction[:, i]
                cos_sim = torch.dot(vi, vj) / (vi.norm() * vj.norm() + 1e-12)
                theta = torch.acos(torch.clamp(cos_sim, -1.0, 1.0)) * 180.0 / torch.pi
                cumulative_muon_delta_angle.append(cumulative_muon_delta_angle[-1] + theta.item())

    return net_adam, net_muon, traj_adam, traj_muon, cumulative_adam_delta_angle, cumulative_muon_delta_angle


def evaluate_models(net_adam, net_muon, repeats=100):
    """Compute average test loss for both models."""
    running_adam_loss, running_muon_loss = 0.0, 0.0
    for i in range(repeats):
        torch.manual_seed(i)
        X_test = torch.randn(1000, 3)
        y_test = system_poly(X_test)
        running_adam_loss += ((net_adam(X_test) - y_test) ** 2).mean().item()
        running_muon_loss += ((net_muon(X_test) - y_test) ** 2).mean().item()
    print("Average Adam loss:", running_adam_loss / repeats)
    print("Average Muon loss:", running_muon_loss / repeats)


def plot_trajectories(traj_adam, traj_muon, save_path="trajectories.png"):
    """Plot Adam and Muon trajectories in 3D."""
    fig = plt.figure(figsize=(10, 5))
    ax1, ax2 = fig.add_subplot(121, projection="3d"), fig.add_subplot(122, projection="3d")

    def plot_traj(ax, traj, title):
        origin, colors = [0, 0, 0], ["r", "g", "b"]
        for step, mat in enumerate(traj):
            for i in range(3):
                v = mat[i]
                ax.plot([origin[0], v[0]], [origin[1], v[1]], [origin[2], v[2]],
                        color=colors[i], alpha=0.3 + 0.7 * (step / len(traj)))
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        ax.set_zlim(-1.5, 1.5)
        ax.set_title(title)

    plot_traj(ax1, traj_adam, "Adam")
    plot_traj(ax2, traj_muon, "Muon")
    plt.savefig(save_path, dpi=150)


def save_animation(traj_adam, traj_muon, nb_frames=100, out_dir="frames"):
    """Save frame-by-frame evolution of trajectories."""
    os.makedirs(out_dir, exist_ok=True)
    origin, colors = [0, 0, 0], ["r", "g", "b"]

    for step in range(min(nb_frames, len(traj_adam))):
        fig = plt.figure(figsize=(10, 5))
        ax1, ax2 = fig.add_subplot(121, projection="3d"), fig.add_subplot(122, projection="3d")

        for i in range(3):
            v = traj_adam[step][i]
            ax1.plot([origin[0], v[0]], [origin[1], v[1]], [origin[2], v[2]], color=colors[i])
        ax1.set_xlim(-1.5, 1.5)
        ax1.set_ylim(-1.5, 1.5)
        ax1.set_zlim(-1.5, 1.5)
        ax1.set_title("Adam")

        for i in range(3):
            v = traj_muon[step][i]
            ax2.plot([origin[0], v[0]], [origin[1], v[1]], [origin[2], v[2]], color=colors[i])
        ax2.set_xlim(-1.5, 1.5)
        ax2.set_ylim(-1.5, 1.5)
        ax2.set_zlim(-1.5, 1.5)
        ax2.set_title("Muon")

        plt.savefig(f"{out_dir}/frame_{step:03d}.png", dpi=150)
        plt.close()


def main():
    parser = argparse.ArgumentParser(description="Compare Adam vs Muon optimizers.")
    parser.add_argument("--mode", choices=["train", "eval", "plot", "animate"], required=True)
    args = parser.parse_args()

    if args.mode == "train":
        net_adam, net_muon, traj_adam, traj_muon, cadam, cmuon = train_models()
        torch.save({"adam": net_adam.state_dict(), "muon": net_muon.state_dict(),
                    "traj_adam": traj_adam, "traj_muon": traj_muon,
                    "cadam": cadam, "cmuon": cmuon}, "results.pth")

    elif args.mode == "eval":
        data = torch.load("results.pth")
        net_adam, net_muon = SmallNet(), SmallNet()
        net_adam.load_state_dict(data["adam"])
        net_muon.load_state_dict(data["muon"])
        evaluate_models(net_adam, net_muon)

    elif args.mode == "plot":
        data = torch.load("results.pth")
        plot_trajectories(data["traj_adam"], data["traj_muon"])

    elif args.mode == "animate":
        data = torch.load("results.pth")
        save_animation(data["traj_adam"], data["traj_muon"])


if __name__ == "__main__":
    main()
