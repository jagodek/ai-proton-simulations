import os
import time
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import math
from pathlib import Path

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
training_data = np.load("training_data.npz")
normalized_data_dose = training_data["normalized_data_dose"]
normalized_data_fluence_protons = training_data["normalized_data_fluence_protons"]
normalized_data_dlet_protons = training_data["normalized_data_dlet_protons"]
X = training_data["normalized_data_x"]

Y = torch.stack([
    torch.tensor(normalized_data_dose,    dtype=torch.float32),
    torch.tensor(normalized_data_fluence_protons, dtype=torch.float32),
    torch.tensor(normalized_data_dlet_protons,     dtype=torch.float32),
], dim=1).to(device)
X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
n = len(X_tensor)

class ProtonDepthProfileNet(nn.Module):
    def __init__(self, hidden_dim: int = 64, n_segments: int = 400):
        super().__init__()
        self.n_segments = n_segments

        self.trunk = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )

        self.head_dose     = nn.Sequential(nn.Linear(hidden_dim, n_segments), nn.Softplus())
        self.head_fluence  = nn.Sequential(nn.Linear(hidden_dim, n_segments), nn.Softplus())
        self.head_let      = nn.Sequential(nn.Linear(hidden_dim, n_segments), nn.Softplus())

    def forward(self, energy: torch.Tensor) -> torch.Tensor:
        if energy.dim() == 1:
            energy = energy.unsqueeze(-1)

        features = self.trunk(energy)

        dose    = self.head_dose(features)
        fluence = self.head_fluence(features)
        let     = self.head_let(features)

        return torch.stack([dose, fluence, let], dim=1)


model     = ProtonDepthProfileNet(16).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=20, factor=0.5)
criterion = nn.MSELoss()

best_val_loss = float("inf")

model.train()
n_samples = X.shape[0]
batch_size = 128
start_training = time.time()
for epoch in range(1000):
    train_loss = 0.0
    perm = torch.randperm(n)
    for i in range(0, n, batch_size):
        idx = perm[i:i+batch_size]
        x_batch, y_batch = X_tensor[idx], Y[idx]

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

os.makedirs('./checkpoints', exist_ok=True)
torch.save(model.state_dict(), './checkpoints/gen3_batch7_ProtonDepthProfileNet_16_dletpreprocessing_02_01.pth')