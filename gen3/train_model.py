import os
import json
import random
import shutil
import re
import math
import time
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from matplotlib.animation import FuncAnimation 
import multiprocessing as mp


db = np.load("training_data.npz")

normalized_data_dose = db["normalized_data_dose"]
normalized_data_fluence_protons = db["normalized_data_fluence_protons"]
# normalized_data_dlet_protons = db[""]
normalized_data_dlet_protons = db["normalized_data_dlet_protons"]
X = db["data_x"]


Y = torch.stack([
    torch.tensor(normalized_data_dose,    dtype=torch.float32),
    torch.tensor(normalized_data_fluence_protons, dtype=torch.float32),
    torch.tensor(normalized_data_dlet_protons,     dtype=torch.float32),
], dim=1)  # → (n_samples, 3, 400)


X_min, X_max = X.min(), X.max()
X_norm = (X - X_min) / (X_max - X_min)

X_tensor = torch.tensor(X_norm, dtype=torch.float32)
n = len(X_norm)



class ProtonDepthProfileNet(nn.Module):
    def __init__(self, hidden_dim: int = 64, n_segments: int = 400):
        super().__init__()
        self.n_segments = n_segments

        # Shared trunk: encodes energy into a rich latent representation
        self.trunk = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )

        # Separate heads — each learns its own physics independently
        self.head_dose     = nn.Sequential(nn.Linear(hidden_dim, n_segments), nn.Softplus())
        self.head_fluence  = nn.Sequential(nn.Linear(hidden_dim, n_segments), nn.Softplus())
        self.head_let      = nn.Sequential(nn.Linear(hidden_dim, n_segments), nn.Softplus())

    def forward(self, energy: torch.Tensor) -> torch.Tensor:
        # energy: (batch_size,) — raw MeV values
        if energy.dim() == 1:
            energy = energy.unsqueeze(-1)         # → (batch, 1)

        features = self.trunk(energy)             # → (batch, hidden_dim)

        dose    = self.head_dose(features)        # → (batch, 400)
        fluence = self.head_fluence(features)     # → (batch, 400)
        let     = self.head_let(features)         # → (batch, 400)

        return torch.stack([dose, fluence, let], dim=1)  # → (batch, 3, 400)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(device)

model     = ProtonDepthProfileNet(16).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=20, factor=0.5)
criterion = nn.MSELoss()

best_val_loss = float("inf")

model.train()
n_samples = X.shape[0]
batch_size = 10
for epoch in range(1000):
    # --- Training ---
    train_loss = 0.0
    perm = torch.randperm(n)
    for i in range(0, n, batch_size):
        idx = perm[i:i+batch_size]
        x_batch, y_batch = X_tensor[idx], Y[idx]
        # print(x_batch)
        # print(y_batch)
        optimizer.zero_grad()
        pred = model(x_batch)           # (batch, 3, 400)
        loss = criterion(pred, y_batch)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    if epoch % 50 == 0:
        # print(f"{model.trunk[0].weight}".replace("\n","").replace(" ",""))
        # print(f"{model.head_dose[0].weight}".replace("\n","").replace(" ",""))
        # print(f"{model.head_dose[2].weight}".replace("\n","").replace(" ",""))
        print(train_loss)


os.makedirs('./checkpoints', exist_ok=True)
torch.save(model.state_dict(), './checkpoints/gen3_batch7_ProtonDepthProfileNet_16_dletpreprocessing_02_01.pth')