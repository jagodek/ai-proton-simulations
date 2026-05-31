import time
import numpy as np
import torch
import torch.nn as nn
import math
import os
import sys
from pathlib import Path
import shutil
slurm_job_id = ""
if len(sys.argv) == 2:
    slurm_job_id = str(sys.argv[1])

HOME = "/home/michal/slrm/gen4"
if os.getenv("PLG_GROUPS_STORAGE"):
    HOME = "/net/people/plgrid/plgmichalgodek/workspace/ai-proton-simulations/gen4"
LOGS_PATH = Path(HOME, "tmp", "logs") 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
training_data = np.load(Path(HOME,"training_data_g4_batch1_dlet_z_processed.npz"))
data_z_dose = training_data["data_z_dose"]
data_z_fluence_protons = training_data["data_z_fluence_protons"]
data_z_dlet_protons = training_data["data_z_dlet_protons"]
data_r_dose = training_data["data_r_dose"]
data_r_fluence_protons = training_data["data_r_fluence_protons"]
data_r_dlet_protons = training_data["data_r_dlet_protons"]
data_x = training_data["data_x"]

print(f"Training on energies [{data_x[0],data_x[1]}...{data_x[-2],data_x[-1]}")
x_min, x_max = np.min(data_x), np.max(data_x)

# per-series maxima
max_z_dose = np.max(data_z_dose)
max_r_dose = np.max(data_r_dose)

max_z_fluence_protons = np.max(data_z_fluence_protons)
max_r_fluence_protons = np.max(data_r_fluence_protons)

max_z_dlet_protons = np.max(data_z_dlet_protons)
max_r_dlet_protons = np.max(data_r_dlet_protons)

normalized_x = (data_x - x_min) / (x_max-x_min)

# normalized per-series
normalized_data_z_dose = data_z_dose / max_z_dose
normalized_data_r_dose = data_r_dose / max_r_dose

normalized_data_z_fluence_protons = data_z_fluence_protons / max_z_fluence_protons
normalized_data_r_fluence_protons = data_r_fluence_protons / max_r_fluence_protons

normalized_data_z_dlet_protons = data_z_dlet_protons / max_z_dlet_protons
normalized_data_r_dlet_protons = data_r_dlet_protons / max_r_dlet_protons


# test_data = np.load(Path(HOME, "test_data_g3batch10.npz"))
# data_z_dose_test = test_data["data_z_dose_test"]
# data_z_fluence_protons_test = test_data["data_z_fluence_protons_test"]
# data_z_dlet_protons_test = test_data["data_z_dlet_protons_test"]
# data_r_dose_test = test_data["data_r_dose_test"]
# data_r_fluence_protons_test = test_data["data_r_fluence_protons_test"]
# data_r_dlet_protons_test = test_data["data_r_dlet_protons_test"]
# data_x_test = test_data["data_x_test"]

# normalized_x_test = (data_x_test - x_min) / (x_max-x_min)
# normalized_data_z_dose_test = data_z_dose_test / max_z_dose
# normalized_data_r_dose_test = data_r_dose_test / max_r_dose
# normalized_data_z_fluence_protons_test = data_z_fluence_protons_test / max_z_fluence_protons
# normalized_data_r_fluence_protons_test = data_r_fluence_protons_test / max_r_fluence_protons
# normalized_data_z_dlet_protons_test = data_z_dlet_protons_test / max_z_dlet_protons
# normalized_data_r_dlet_protons_test = data_r_dlet_protons_test / max_r_dlet_protons      

# with open(LOGS_PATH,"w+") as f:
#     pass

# def test_model(model, criterion, device):
#     """
#     Evaluates the trained model on the test dataset.
#     """
#     model.eval()
#     eps = 10e-9


#     normalized_data_z_dose_test_wholes, normalized_data_z_dose_test_halves = normalized_data_z_dose_test[::2], normalized_data_z_dose_test[1::2]
#     normalized_data_z_fluence_protons_test_wholes, normalized_data_z_fluence_protons_test_halves = normalized_data_z_fluence_protons_test[::2], normalized_data_z_fluence_protons_test[1::2]
#     normalized_data_z_dlet_protons_test_wholes, normalized_data_z_dlet_protons_test_halves = normalized_data_z_dlet_protons_test[::2], normalized_data_z_dlet_protons_test[1::2]
#     data_x_test_wholes, data_x_test_halves = data_x_test[::2], data_x_test[1::2]
#     normalized_data_r_dose_test_wholes, normalized_data_r_dose_test_halves = normalized_data_r_dose_test[::2], normalized_data_r_dose_test[1::2]
#     normalized_data_r_fluence_protons_test_wholes, normalized_data_r_fluence_protons_test_halves = normalized_data_r_fluence_protons_test[::2], normalized_data_r_fluence_protons_test[1::2]
#     normalized_data_r_dlet_protons_test_wholes, normalized_data_r_dlet_protons_test_halves = normalized_data_r_dlet_protons_test[::2], normalized_data_r_dlet_protons_test[1::2]

#     Y_z_test_wholes = torch.stack(
#         [
#             torch.tensor(normalized_data_z_dose_test_wholes, dtype=torch.float32),
#             torch.tensor(normalized_data_z_fluence_protons_test_wholes, dtype=torch.float32),
#             torch.tensor(normalized_data_z_dlet_protons_test_wholes, dtype=torch.float32),
#         ],
#         dim=1,
#     ).to(device)

#     Y_z_test_halves = torch.stack(
#         [
#             torch.tensor(normalized_data_z_dose_test_halves, dtype=torch.float32),
#             torch.tensor(normalized_data_z_fluence_protons_test_halves, dtype=torch.float32),
#             torch.tensor(normalized_data_z_dlet_protons_test_halves, dtype=torch.float32),
#         ],
#         dim=1,
#     ).to(device)


#     Y_r_test_wholes = torch.stack(
#         [
#             torch.tensor(normalized_data_r_dose_test_wholes, dtype=torch.float32),
#             torch.tensor(normalized_data_r_fluence_protons_test_wholes, dtype=torch.float32),
#             torch.tensor(normalized_data_r_dlet_protons_test_wholes, dtype=torch.float32),
#         ],
#         dim=1,
#     ).to(device)

#     Y_r_test_halves = torch.stack(
#         [
#             torch.tensor(normalized_data_r_dose_test_halves, dtype=torch.float32),
#             torch.tensor(normalized_data_r_fluence_protons_test_halves, dtype=torch.float32),
#             torch.tensor(normalized_data_r_dlet_protons_test_halves, dtype=torch.float32),
#         ],
#         dim=1,
#     ).to(device)

#     X_test_wholes_tensor = torch.tensor(data_x_test_wholes, dtype=torch.float32).to(device)
#     X_test_halves_tensor = torch.tensor(data_x_test_halves, dtype=torch.float32).to(device)

#     with torch.no_grad():
#         predictions_wholes = model(X_test_wholes_tensor)
#         sq_err_z = (predictions_wholes - Y_z_test_wholes) / (Y_z_test_wholes.abs() + eps)
#         final_test_loss_z_wholes = (sq_err_z**2).mean()   

#         sq_err_r = (predictions_wholes - Y_r_test_wholes) / (Y_r_test_wholes.abs() + eps)
#         final_test_loss_r_wholes = (sq_err_r**2).mean()


#         predictions_halves = model(X_test_halves_tensor)
#         sq_err_z = (predictions_halves - Y_z_test_halves) / (Y_z_test_halves.abs() + eps)
#         final_test_loss_z_halves = (sq_err_z**2).mean()

#         sq_err_z = (predictions_halves - Y_z_test_halves) / (Y_z_test_halves.abs() + eps)
#         final_test_loss_z_halves = (sq_err_z**2).mean()
#     #     weighted_test_loss_halves = (sq_err_batch**2) * weights_testor_test_halves

#     # with torch.no_grad():

#     #     predictions_wholes = model(X_test_wholes_tensor)
#     #     test_loss_wholes = criterion(predictions_wholes, Y_test_wholes).item()

#     #     predictions_halves = model(X_test_halves_tensor)
#     #     test_loss_halves = criterion(predictions_halves, Y_test_halves).item()
#     criterion_name = criterion.__class__.__name__
#     print(f"\n--- Model Evaluation ---")
#     print(f"Wholes Test {criterion_name}: {test_loss_wholes:.4e}")
#     print(f"Halves Test {criterion_name}: {test_loss_halves:.4e}")
#     with open(LOGS_PATH, "a+") as logs_file:
#         logs_file.write(f"Wholes Test {criterion_name}: {test_loss_wholes:.4e}\n")
#         logs_file.write(f"Halves Test {criterion_name}: {test_loss_halves:.4e}\n")
#     return  test_loss_wholes, test_loss_halves

seeds_per_energy = 0
for i,x in enumerate(data_x.reshape(-1)):
    if float(x) == data_x[0]:
        seeds_per_energy += 1
    else:
        break
print(f"Seeds per energy: {seeds_per_energy}")

Y_z = torch.stack([
    torch.tensor(normalized_data_z_dose, dtype=torch.float32),
    torch.tensor(normalized_data_z_fluence_protons, dtype=torch.float32),
    torch.tensor(normalized_data_z_dlet_protons, dtype=torch.float32),
], dim=1).to(device)

Y_r =torch.stack([
    torch.tensor(normalized_data_r_dose, dtype=torch.float32),
    torch.tensor(normalized_data_r_fluence_protons, dtype=torch.float32),
    torch.tensor(normalized_data_r_dlet_protons, dtype=torch.float32),
], dim=1).to(device)

X_tensor = torch.tensor(normalized_x, dtype=torch.float32).to(device)
n_samples = n_samples = len(X_tensor)

def loss_fn(pred, target):
    # target is a tuple: (Y_z_batch, Y_r_batch)
    # Y_z_batch shape: (batch, 3, n_z), Y_r_batch shape: (batch, 3, n_r)
    y_z, y_r = target

    dose_z_target = y_z[:, 0, :]
    fluence_z_target = y_z[:, 1, :]
    dlet_z_target = y_z[:, 2, :]
    
    dose_r_target = y_r[:, 0, :]
    fluence_r_target = y_r[:, 1, :]
    dlet_r_target = y_r[:, 2, :]

    loss_dose_z = ((pred["dose_z"] - dose_z_target) ** 2).mean()
    loss_fluence_z = ((pred["fluence_z"] - fluence_z_target) ** 2).mean()
    loss_dlet_z = ((pred["dlet_z"] - dlet_z_target) ** 2).mean()
    
    loss_dose_r = ((pred["dose_r"] - dose_r_target) ** 2).mean()
    loss_fluence_r = ((pred["fluence_r"] - fluence_r_target) ** 2).mean()
    loss_dlet_r = ((pred["dlet_r"] - dlet_r_target) ** 2).mean()
    
    return (
        loss_dose_z +
        loss_fluence_z +
        loss_dlet_z +
        loss_dose_r +
        loss_fluence_r +
        loss_dlet_r
    )


class Model(nn.Module):
    def __init__(self, hidden_dim=128):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_z = 400
        self.n_r = 100

        self.trunk = nn.Sequential(
            nn.Linear(1, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.SiLU(),
        )

        # depth heads
        self.head_dose_z = nn.Sequential(
            nn.Linear(self.hidden_dim, self.n_z),
            nn.Softplus()
        )
        self.head_fluence_z = nn.Sequential(
            nn.Linear(self.hidden_dim, self.n_z),
            nn.Softplus()
        )
        self.head_dlet_z = nn.Sequential(
            nn.Linear(self.hidden_dim, self.n_z),
            nn.Softplus()
        )

        # lateral heads
        self.head_dose_r = nn.Linear(self.hidden_dim, self.n_r)
        self.head_fluence_r = nn.Linear(self.hidden_dim, self.n_r)
        self.head_dlet_r = nn.Sequential(
            nn.Linear(self.hidden_dim, self.n_r),
            nn.Softplus()
        )

    def forward(self, energy: torch.Tensor):
        if energy.dim() == 1:
            energy = energy.unsqueeze(-1)

        features = self.trunk(energy)

        dose_z = self.head_dose_z(features)
        fluence_z = self.head_fluence_z(features)
        dlet_z = self.head_dlet_z(features)

        dose_r = self.head_dose_r(features)
        fluence_r = self.head_fluence_r(features)
        dlet_r = self.head_dlet_r(features)

        return {
            "dose_z": dose_z,
            "fluence_z": fluence_z,
            "dlet_z": dlet_z,
            "dose_r": dose_r,
            "fluence_r": fluence_r,
            "dlet_r": dlet_r,
        }

batch_size = 128
total_epochs = 1000

model = Model().to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5000)
criterion = nn.MSELoss()
start_training = time.time()

model.train()
start_training = time.time()
for epoch in range(total_epochs):
    train_loss = 0.0
    perm = torch.randperm(n_samples)
    for i in range(0, n_samples, batch_size):
        idx = perm[i:i+batch_size]
        x_batch, y_batch = X_tensor[idx], (Y_z[idx], Y_r[idx])

        optimizer.zero_grad()
        pred = model(x_batch)
        loss = loss_fn(pred, y_batch)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    num_batches = math.ceil(n_samples / batch_size)
    train_loss /= num_batches
    current_lr = optimizer.param_groups[0]['lr']

    if epoch % 50 == 0:
        print(
            f"Epoch {epoch:>4d} | ",
            f"Train loss: {train_loss:.4e} | ",
            f"LR: {current_lr:.2e}",
            f"Time: {time.time()-start_training:.2f}"
        )


# final_test_loss_wholes, final_test_loss_halves = test_model(model, criterion, device)

CHECKPOINTS_DIR_NAME = "checkpoints"+slurm_job_id
if not Path(HOME, CHECKPOINTS_DIR_NAME).is_dir():
    Path(HOME, CHECKPOINTS_DIR_NAME).mkdir()

# def save_checkpoints():
#     with open(Path(HOME,CHECKPOINTS_DIR_NAME, "best_losses_history"), "a") as best_losses_history:
#         best_losses_history.write(f"{final_test_loss_wholes},{final_test_loss_halves}\n")
#     with open(Path(HOME,CHECKPOINTS_DIR_NAME, "best_losses_history"), "r") as best_losses_history:
#         losses_number = len(best_losses_history.readlines())-1
#     with open(Path(HOME, CHECKPOINTS_DIR_NAME, "best_loss"), "w") as best_loss_file:
#         best_loss_file.write(f"{final_test_loss_wholes},{final_test_loss_halves}")
#         # os.makedirs('./checkpoints', exist_ok=True)
#     with open(Path(HOME, CHECKPOINTS_DIR_NAME, "best_code"+str(losses_number)), "w") as best_code_file:
#         with open(Path(HOME,"tmp","train_model_loop.py"), "r") as current_code_file:
#             best_code_file.write(current_code_file.read())

# torch.save(model.state_dict(), Path(HOME,CHECKPOINTS_DIR_NAME,f"basic{slurm_job_id}.pth"))
torch.save(model, Path(HOME,CHECKPOINTS_DIR_NAME,f"basic_model{slurm_job_id}.pth"))
shutil.copy(Path(HOME, "train_model.py"), Path(HOME,CHECKPOINTS_DIR_NAME, "train_model.py"))

# alpha = 0.75
# if not Path(HOME, CHECKPOINTS_DIR_NAME).is_dir():
#     Path(HOME, CHECKPOINTS_DIR_NAME).mkdir()
#     with open(Path(HOME, CHECKPOINTS_DIR_NAME, "best_losses_history"),"w"):
#         pass
#     Path(HOME, CHECKPOINTS_DIR_NAME, "")
#     save_checkpoints()
# else:
#     with open(Path(HOME, CHECKPOINTS_DIR_NAME, "best_loss"), "r") as best_loss_file:
#         best_loss_tuple = best_loss_file.readline()
#         best_loss_wholes, best_loss_halves = best_loss_tuple.split(",")
#         best_loss_wholes, best_loss_halves = float(best_loss_wholes), float(best_loss_halves) 

#         combined_best_loss = alpha * best_loss_wholes + (1 - alpha) * best_loss_halves
#         combined_loss = alpha * final_test_loss_wholes + (1 - alpha) * final_test_loss_halves
        
#     if combined_loss < combined_best_loss:
#         save_checkpoints()
