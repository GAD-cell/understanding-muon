import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class SmallNet(nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(10)
        self.linear = nn.Linear(3, 3, bias=False) 
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.linear(x))


# X = torch.eye(3)
# y = torch.eye(3)
torch.manual_seed(45)
coeffs = {
    "p1": [1.0, 0.5, -0.3, 0.2, 0.1, -0.2, 0.3, -0.1, 0.0, 0.0],
    "p2": [0.5, -0.2, 0.4, 0.1, -0.3, 0.2, -0.1, 0.2, 0.0, 0.1],
    "p3": [-0.3, 0.4, 0.1, -0.2, 0.3, -0.1, 0.2, -0.2, 0.0, -0.1]
}

def poly_eval(x, coefs):
    x1, x2, x3 = x[:,0], x[:,1], x[:,2]
    a,b,c,d,e,f,g,h,i,j = coefs
    return (a*x1**2 + b*x2**2 + c*x3**2 +
            d*x1*x2 + e*x1*x3 + f*x2*x3 +
            g*x1 + h*x2 + i*x3 + j)

def system_poly(X):
    f1 = poly_eval(X, coeffs["p1"])
    f2 = poly_eval(X, coeffs["p2"])
    f3 = poly_eval(X, coeffs["p3"])
    return torch.stack([f1,f2,f3], dim=1)  # shape = (batch,3)

X = torch.randn(100,3)
y = system_poly(X)

net_adam = SmallNet()
net_muon = SmallNet()

optim_adam = optim.Adam(net_adam.parameters(), lr=0.005)
from muon import MuonClip, MuonConfig
muon_config = MuonConfig(muon_lr=0.05,better_ortho=False,ns_steps=5,enable_clipping=False,log_dir="")
optim_muon = MuonClip(net_muon, {}, muon_config)


basis = torch.eye(3)
traj_adam = []
traj_muon = []
sims_adam = []
sims_muon = []
import itertools
for step in range(1000):

    optim_adam.zero_grad()
    loss_adam = ((net_adam(X) - y)**2).mean()
    loss_adam.backward()
    optim_adam.step()
    with torch.no_grad():
        transformed = net_adam.linear.weight @ basis.T 
        V = transformed.T                             
        traj_adam.append(V.numpy())

        mean_theta = 0.0
        for (i, j) in itertools.combinations(range(3), 2):
            vi, vj = V[:, i], V[:, j]
            cos_sim = torch.dot(vi, vj) / (vi.norm() * vj.norm() + 1e-12)
            theta = torch.acos(torch.clamp(cos_sim, -1.0, 1.0)) * 180.0 / torch.pi
            mean_theta += theta.item() 
        sims_adam.append(mean_theta / 3.0)

    optim_muon.zero_grad()
    loss_muon = ((net_muon(X) - y)**2).mean()
    loss_muon.backward()
    optim_muon.step()
    with torch.no_grad():
        transformed = net_muon.linear.weight @ basis.T
        V = transformed.T                             
        traj_muon.append(V.numpy())

        mean_theta = 0.0
        for (i, j) in itertools.combinations(range(3), 2):
            vi, vj = V[:, i], V[:, j]
            cos_sim = torch.dot(vi, vj) / (vi.norm() * vj.norm() + 1e-12)
            theta = torch.acos(torch.clamp(cos_sim, -1.0, 1.0)) * 180.0 / torch.pi
            mean_theta += theta.item()  
        sims_muon.append(mean_theta / 3.0)


# plot similarities
plt.figure(figsize=(8,5))
plt.plot(sims_adam, label="Adam", color="blue")
plt.plot(sims_muon, label="Muon", color="orange")
plt.xlabel("Training step")
plt.ylabel("Mean angle (degrees)")
plt.legend()
plt.grid()
plt.savefig("similarities.png", dpi=150)
plt.close()

running_adam_loss = 0.0
running_muon_loss = 0.0
for i in range(100):
    torch.manual_seed(i)
    X_test = torch.randn(1000,3)
    y_test = system_poly(X_test)
    running_adam_loss += ((net_adam(X_test) - y_test)**2).mean().item()
    running_muon_loss += ((net_muon(X_test) - y_test)**2).mean().item()
print("Average Adam loss:", running_adam_loss / 100)
print("Average Muon loss:", running_muon_loss / 100)    


fig = plt.figure(figsize=(10, 5))
ax = fig.add_subplot(121, projection="3d")
ax2 = fig.add_subplot(122, projection="3d")

def plot_trajectory(ax, traj, title):
    origin = [0, 0, 0]
    colors = ["r", "g", "b"]

    for step, mat in enumerate(traj):
        for i in range(3):
            v = mat[i]  # vecteur transformé
            ax.plot([origin[0], v[0]], [origin[1], v[1]], [origin[2], v[2]],
                    color=colors[i], alpha=0.3 + 0.7*(step/len(traj)))
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_zlim(-1.5, 1.5)
    ax.set_title(title)

nb_frames = 50
plot_trajectory(ax, traj_adam, "Adam")
plot_trajectory(ax2, traj_muon, "Muon")

plt.savefig("trajectoires.png", dpi=150)

import os
import matplotlib.pyplot as plt

os.makedirs("frames", exist_ok=True)

for step in range(len(traj_adam[:nb_frames])):
    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(121, projection="3d")
    ax2 = fig.add_subplot(122, projection="3d")

    origin = [0,0,0]
    colors = ["r","g","b"]

    mat = traj_adam[step]
    for i in range(3):
        v = mat[i]
        ax.plot([origin[0], v[0]], [origin[1], v[1]], [origin[2], v[2]],
                color=colors[i])
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_zlim(-1.5, 1.5)
    ax.set_title("Adam")

    mat = traj_muon[step]
    for i in range(3):
        v = mat[i]
        ax2.plot([origin[0], v[0]], [origin[1], v[1]], [origin[2], v[2]],
                 color=colors[i])
    ax2.set_xlim(-1.5, 1.5)
    ax2.set_ylim(-1.5, 1.5)
    ax2.set_zlim(-1.5, 1.5)
    ax2.set_title("Muon")

    plt.savefig(f"frames/frame_{step:03d}.png", dpi=150)
    plt.close()