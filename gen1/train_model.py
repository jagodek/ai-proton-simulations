import os
import time
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import math
from pathlib import Path

slurm_job_id = ""
if len(sys.argv) == 2:
    slurm_job_id = str(sys.argv[1])
HOME = "/home/michal/slrm/gen1"
if os.getenv("PLG_GROUPS_STORAGE"):
    HOME = "/net/people/plgrid/plgmichalgodek/workspace/ai-proton-simulations/gen1"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
training_data = np.load("training_data_g1batch1_2.npz")
data_dose = training_data["data_dose"]
data_x = training_data["data_x"]
max_dose = np.max(data_dose)
x_min, x_max = np.min(data_x), np.max(data_x)
normalized_x = (data_x - x_min) / (x_max-x_min)

normalized_data_z_dose = data_dose / max_dose
X_tensor = torch.from_numpy(normalized_x.astype(np.float32)).to(device)
Y_tensor = torch.from_numpy(normalized_data_z_dose.astype(np.float32)).to(device)

n = len(X_tensor)


class Model(nn.Module):
    def __init__(self):
        self.hidden_dim = 128
        self.n_depths=400
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(1, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.n_depths),
            nn.Softplus()  
        )
    
    def forward(self, energy):
        return self.net(energy.unsqueeze(-1)).squeeze(-1)


model     = Model().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()

best_val_loss = float("inf")

model.train()
n_samples = X_tensor.shape[0]
batch_size = 128
start_training = time.time()
for epoch in range(1000):
    train_loss = 0.0
    perm = torch.randperm(n)
    for i in range(0, n, batch_size):
        idx = perm[i:i+batch_size]
        x_batch, y_batch = X_tensor[idx], Y_tensor[idx]

        optimizer.zero_grad()
        pred = model(x_batch)
        loss = criterion(pred, y_batch)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    num_batches = math.ceil(n / batch_size)
    train_loss /= num_batches
    current_lr = optimizer.param_groups[0]['lr']
    if epoch % 50 == 0:
        print(
            f"Epoch {epoch:>4d} | ",
            f"Train: {train_loss:.4e} | ",
            f"LR: {current_lr:.2e}",
            f"Time: {time.time()-start_training:.2f}"
        )

CHECKPOINTS_DIRNAME = "checkpoints"+slurm_job_id
Path.mkdir(Path(HOME, CHECKPOINTS_DIRNAME),exist_ok=True)
MODEL_FILENAME = "model"
if Path(HOME,CHECKPOINTS_DIRNAME, MODEL_FILENAME+".pth").exists():
    MODEL_FILENAME += str(time())

torch.save(model, Path(HOME,CHECKPOINTS_DIRNAME, "model.pth"))

