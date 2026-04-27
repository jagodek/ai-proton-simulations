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
print(f"Device: {device}")
training_data = np.load("training_data.npz")
normalized_data_dose = training_data["normalized_data_dose"]
normalized_data_fluence_protons = training_data["normalized_data_fluence_protons"]
normalized_data_dlet_protons = training_data["normalized_data_dlet_protons"]
X = training_data["normalized_data_x"]

validation_part = 0.2

seeds_per_energy = 0
for i,x in enumerate(X):
    if x == X[0]:
        seeds_per_energy += 1
    else:
        break
print(seeds_per_energy)

n_validation = math.floor(validation_part*seeds_per_energy)

mask = np.zeros(len(X), dtype=bool)
for i in range(n_validation):
    mask[i::seeds_per_energy] = True
    mask[i::seeds_per_energy] = True

X_val, normalized_data_dose_val, normalized_data_fluence_protons_val, normalized_data_dlet_protons_val = X[mask], normalized_data_dose[mask], normalized_data_fluence_protons[mask], normalized_data_dlet_protons[mask]

mask = np.zeros(len(X), dtype=bool)
for i in range(n_validation,seeds_per_energy):
    mask[i::seeds_per_energy] = True

X, normalized_data_dose, normalized_data_fluence_protons, normalized_data_dlet_protons = X[mask], normalized_data_dose[mask], normalized_data_fluence_protons[mask], normalized_data_dlet_protons[mask]



Y_val = torch.stack([
    torch.tensor(normalized_data_dose_val,            dtype=torch.float32),
    torch.tensor(normalized_data_fluence_protons_val, dtype=torch.float32),
    torch.tensor(normalized_data_dlet_protons_val,    dtype=torch.float32),
], dim=1).to(device)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)
n_val = len(X_val_tensor)


Y = torch.stack([
    torch.tensor(normalized_data_dose,    dtype=torch.float32),
    torch.tensor(normalized_data_fluence_protons, dtype=torch.float32),
    torch.tensor(normalized_data_dlet_protons,     dtype=torch.float32),
], dim=1).to(device)
X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
n = len(X_tensor)

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.n_segments = 400
        self.hidden_dim = 64

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


model     = Model().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=20, factor=0.5)
criterion = nn.MSELoss()

best_val_loss = float("inf")

model.train()
n_samples = X.shape[0]
batch_size = 128
total_epochs = 1000
start_training = time.time()
for epoch in range(total_epochs):
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
    
    
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for i in range(0, n_val, batch_size):
            x_batch = X_val_tensor[i:i + batch_size]
            y_batch = Y_val[i:i + batch_size]
            pred = model(x_batch)
            val_loss += criterion(pred, y_batch).item()

    num_val_batches = math.ceil(n_val / batch_size)
    val_loss /= num_val_batches

    scheduler.step(val_loss)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(
            model.state_dict(),
            './checkpoints/gen3_batch7_ProtonDepthProfileNet_16_dletpreprocessing_02_01_best.pth'
        )


    if epoch % 50 == 0:
        print(
            f"Epoch {epoch:>4d} | ",
            f"Train: {train_loss:.4e} | ",
            f"LR: {current_lr:.2e} | ",
            f"Time: {time.time()-start_training:.2f}"
        )
        with open("logs","a") as logs_file:
            logs_file.write(
                f"Epoch {epoch:>4d} | Train: {train_loss:.4e} | LR: {current_lr:.2e} | Time: {time.time()-start_training:.2f}\n"
            )
        

# os.makedirs('./checkpoints', exist_ok=True)
# torch.save(model.state_dict(), './checkpoints/gen3_batch7_ProtonDepthProfileNet_16_dletpreprocessing_02_01.pth')