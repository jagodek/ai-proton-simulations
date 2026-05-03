import time
import numpy as np
import torch
import torch.nn as nn
import math
import os
import sys
from pathlib import Path

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

with open(LOGS_PATH,"w+") as f:
    pass

def test_model(model, criterion, device, batch_size=128):
    """
    Evaluates the trained model on the test dataset.
    """
    # 1. Set the model to evaluation mode
    model.eval()
    
    # 2. Prepare test tensors and move them to the correct device
    Y_test = torch.stack([
        torch.tensor(normalized_data_dose_test, dtype=torch.float32),
        torch.tensor(normalized_data_fluence_protons_test, dtype=torch.float32),
        torch.tensor(normalized_data_dlet_protons_test, dtype=torch.float32),
    ], dim=1).to(device)
    
    X_test_tensor = torch.tensor(data_x_test, dtype=torch.float32).to(device)
    n_test = len(X_test_tensor)
    
    total_test_loss = 0.0
    
    # 3. Disable gradient computation to save memory and speed up inference
    with torch.no_grad():
        for i in range(0, n_test, batch_size):
            x_batch = X_test_tensor[i:i+batch_size]
            y_batch = Y_test[i:i+batch_size]
            
            # Forward pass
            pred = model(x_batch)
            loss = criterion(pred, y_batch)
            total_test_loss += loss.item()

    
    print(f"\n--- Model Evaluation ---")
    print(f"Final Test Loss: {total_test_loss:.4e}")
    with open(LOGS_PATH,"a+") as logs_file:
            logs_file.write(
                f"Final Test Loss: {total_test_loss:.4e}"
            )
    return total_test_loss


Y = torch.stack([
    torch.tensor(normalized_data_dose,    dtype=torch.float32),
    torch.tensor(normalized_data_fluence_protons, dtype=torch.float32),
    torch.tensor(normalized_data_dlet_protons,     dtype=torch.float32),
], dim=1).to(device)
X_tensor = torch.tensor(normalized_x, dtype=torch.float32).to(device)
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

model.train()
n_samples = data_x.shape[0]
batch_size = 128
total_epochs = 300
start_training = time.time()

for epoch in range(total_epochs):
    # training_loop_placeholder_start
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
    # training_loop_placeholder_end



    if epoch % 50 == 0:
        print(
            f"Epoch {epoch:>4d} | ",
            f"Train loss: {train_loss:.4e} | ",
            f"LR: {current_lr:.2e}",
            f"Time: {time.time()-start_training:.2f}"
        )
        with open(LOGS_PATH,"a+") as logs_file:
            logs_file.write(
                f"Epoch {epoch:>4d} | Train loss: {train_loss:.8e}  | LR: {current_lr:.2e} | Time: {time.time()-start_training:.2f}\n"
            )

final_test_loss = test_model(model, criterion, device, batch_size=128)


def save_checkpoints():
    with open(Path(HOME, "checkpoints", "best_loss"), "w") as best_loss_file:
        best_loss_file.write(str(final_test_loss))
        # os.makedirs('./checkpoints', exist_ok=True)
    with open(Path(HOME, "checkpoints", "best_code"), "w") as best_code_file:
        with open(Path(HOME,"tmp","train_model_loop.py"), "r") as current_code_file:
            best_code_file.write(current_code_file.read())

    torch.save(model.state_dict(), Path(HOME,"checkpoints","best.pth"))
    torch.save(model, Path(HOME,"checkpoints","best_model.pth"))



if not Path(HOME, "checkpoints", "best_loss").is_dir():
    save_checkpoints()
else:
    with open(Path(HOME, "checkpoints", "best_loss"), "r") as best_loss_file:
        best_loss = float(best_loss_file.readline())

    if final_test_loss < best_loss:
        save_checkpoints()

