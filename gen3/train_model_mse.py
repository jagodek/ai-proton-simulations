import os
import time
import numpy as np
import torch
import torch.nn as nn
import math
import os
import sys
from pathlib import Path
slurm_job_id = ""
if len(sys.argv) == 2:
    slurm_job_id = str(sys.argv[1])

HOME = "/home/michal/slrm/gen3/autosearch"
if os.getenv("PLG_GROUPS_STORAGE"):
    HOME = "/net/people/plgrid/plgmichalgodek/workspace/ai-proton-simulations/gen3/autosearch"
LOGS_PATH = Path(HOME, "tmp", "logs") 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
training_data = np.load(Path(HOME,"training_data_g3batch7.npz"))
data_dose = training_data["data_dose"]
data_fluence_protons = training_data["data_fluence_protons"]
data_dlet_protons = training_data["data_dlet_protons"]
data_x = training_data["data_x"]
print(f"Training on energies [{data_x[0],data_x[1]}...{data_x[-2],data_x[-1]}")
x_min, x_max = np.min(data_x), np.max(data_x)
max_dose = np.max(data_dose)
max_fluence_protons = np.max(data_fluence_protons)
max_dlet_protons = np.max(data_dlet_protons)

normalized_x = (data_x - x_min) / (x_max-x_min)
normalized_data_dose = data_dose/max_dose
normalized_data_fluence_protons = data_fluence_protons/max_fluence_protons
normalized_data_dlet_protons = data_dlet_protons/max_dlet_protons


test_data = np.load(Path(HOME, "test_data_g3batch10.npz"))
data_dose_test = test_data["data_dose_test"]
data_fluence_protons_test = test_data["data_fluence_protons_test"]
data_dlet_protons_test = test_data["data_dlet_protons_test"]
data_x_test = test_data["data_x_test"]

normalized_x_test = (data_x_test - x_min) / (x_max-x_min)
normalized_data_dose_test = data_dose_test/max_dose
normalized_data_fluence_protons_test = data_fluence_protons_test/max_fluence_protons
normalized_data_dlet_protons_test = data_dlet_protons_test/max_dlet_protons      


def proportional_mse_loss(pred, target):
    eps = 1e-9
    sq_err_batch = (pred - target) / (target.abs() + eps)

    weighted = (sq_err_batch**2)

    return weighted.mean()

Y = torch.stack([
    torch.tensor(normalized_data_dose,    dtype=torch.float32),
    torch.tensor(normalized_data_fluence_protons, dtype=torch.float32),
    torch.tensor(normalized_data_dlet_protons,     dtype=torch.float32),
], dim=1).to(device)
X_tensor = torch.tensor(normalized_x, dtype=torch.float32).to(device)
n_samples = n_samples = len(X_tensor)

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.n_segments = 400
        self.hidden_dim = 128

        self.trunk = nn.Sequential(
            nn.Linear(1, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.SiLU(),
        )

        self.head_dose     = nn.Sequential(nn.Linear(self.hidden_dim, self.n_segments), nn.Softplus())
        self.head_fluence  = nn.Sequential(nn.Linear(self.hidden_dim, self.n_segments), nn.Softplus())
        self.head_let      = nn.Sequential(nn.Linear(self.hidden_dim, self.n_segments), nn.Softplus())

    def forward(self, energy: torch.Tensor) -> torch.Tensor:
        if energy.dim() == 1:
            energy = energy.unsqueeze(-1)

        features = self.trunk(energy)

        dose    = self.head_dose(features)
        fluence = self.head_fluence(features)
        let     = self.head_let(features)

        return torch.stack([dose, fluence, let], dim=1)

batch_size = 64
total_epochs = 1000

model     = Model().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=20, factor=0.5)
# criterion = nn.MSELoss()

model.train()
start_training = time.time()

for epoch in range(total_epochs):
    train_loss = 0.0
    perm = torch.randperm(n_samples)
    for i in range(0, n_samples, batch_size):
        idx = perm[i:i+batch_size]
        x_batch, y_batch = X_tensor[idx], Y[idx]

        optimizer.zero_grad()
        pred = model(x_batch)
        loss = proportional_mse_loss(pred, y_batch)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    num_batches = math.ceil(n_samples / batch_size)
    train_loss /= num_batches
    current_lr = optimizer.param_groups[0]['lr']
    if epoch % 50 == 0:
        print(
            f"Epoch {epoch:>4d} | ",
            f"Train: {train_loss:.4e} | ",
            f"LR: {current_lr:.2e}",
            f"Time: {time.time()-start_training:.2f}"
        )



os.makedirs('./checkpoints'+slurm_job_id, exist_ok=True)
torch.save(model.state_dict(), './checkpoints/model.pth')

