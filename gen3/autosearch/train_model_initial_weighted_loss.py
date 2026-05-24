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
TMP_DIR = Path(HOME, "tmp"+slurm_job_id)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
training_data = np.load(Path(HOME,"training_data_g3batch7.npz"))
data_dose = training_data["data_dose"]
data_fluence_protons = training_data["data_fluence_protons"]
data_dlet_protons = training_data["data_dlet_protons"]
data_x = training_data["data_x"]

seeds_per_energy = 0
for i,x in enumerate(data_x):
    if float(x) == data_x[0]:
        seeds_per_energy += 1
    else:
        break

print(f"Training on energies [{data_x[0],data_x[1]}...{data_x[-2],data_x[-1]}")
x_min, x_max = np.min(data_x), np.max(data_x)
max_dose = np.max(data_dose)
max_fluence_protons = np.max(data_fluence_protons)
max_dlet_protons = np.max(data_dlet_protons)

normalized_x = (data_x - x_min) / (x_max-x_min)
normalized_data_dose = data_dose/max_dose
normalized_data_fluence_protons = data_fluence_protons/max_fluence_protons
normalized_data_dlet_protons = data_dlet_protons/max_dlet_protons
unique_xs = normalized_x[::seeds_per_energy]

test_data = np.load(Path(HOME, "test_data_g3batch10.npz"))
data_dose_test = test_data["data_dose_test"]
data_fluence_protons_test = test_data["data_fluence_protons_test"]
data_dlet_protons_test = test_data["data_dlet_protons_test"]
data_x_test = test_data["data_x_test"]

normalized_x_test = (data_x_test - x_min) / (x_max-x_min)
normalized_data_dose_test = data_dose_test/max_dose
normalized_data_fluence_protons_test = data_fluence_protons_test/max_fluence_protons
normalized_data_dlet_protons_test = data_dlet_protons_test/max_dlet_protons




reshaped_data_fluence = data_fluence_protons.reshape(-1,seeds_per_energy,  400)
averaged_data_fluence = reshaped_data_fluence.mean(axis=1)
weights_mask_train = np.ones((len(averaged_data_fluence),400))

for i in range(len(averaged_data_fluence)):
    max_fluence_in_i = np.max(averaged_data_fluence[i])
    for k in range(400):
        if averaged_data_fluence[i][k] < 0.001*max_fluence_in_i:
            weights_mask_train[i][:k] = 100
            break
weights_mask_train = weights_mask_train / weights_mask_train.mean(axis=1, keepdims=True)



weights_mask_test = np.ones((len(data_fluence_protons_test),400))

for i in range(len(weights_mask_test)):
    max_fluence_in_i = np.max(data_fluence_protons_test[i])
    for k in range(400):
        if data_fluence_protons_test[i][k] < 0.001*max_fluence_in_i:
            weights_mask_test[i][:k] = 100
            break
weights_mask_test = weights_mask_test / weights_mask_test.mean(axis=1,keepdims=True)




with open(LOGS_PATH,"w+") as f:
    pass

def test_model(model, criterion):
    """
    Evaluates the trained model on the test dataset.
    """
    model.eval()


    normalized_data_dose_test_wholes, normalized_data_dose_test_halves = normalized_data_dose_test[::2], normalized_data_dose_test[1::2]
    normalized_data_fluence_protons_test_wholes, normalized_data_fluence_protons_test_halves = normalized_data_fluence_protons_test[::2], normalized_data_fluence_protons_test[1::2]
    normalized_data_dlet_protons_test_wholes, normalized_data_dlet_protons_test_halves = normalized_data_dlet_protons_test[::2], normalized_data_dlet_protons_test[1::2]
    data_x_test_wholes, data_x_test_halves = data_x_test[::2], data_x_test[1::2]



    Y_test_wholes = torch.stack(
        [
            torch.tensor(normalized_data_dose_test_wholes, dtype=torch.float32),
            torch.tensor(normalized_data_fluence_protons_test_wholes, dtype=torch.float32),
            torch.tensor(normalized_data_dlet_protons_test_wholes, dtype=torch.float32),
        ],
        dim=1,
    ).to(device)

    Y_test_halves = torch.stack(
        [
            torch.tensor(normalized_data_dose_test_halves, dtype=torch.float32),
            torch.tensor(normalized_data_fluence_protons_test_halves, dtype=torch.float32),
            torch.tensor(normalized_data_dlet_protons_test_halves, dtype=torch.float32),
        ],
        dim=1,
    ).to(device)

    X_test_wholes_tensor = torch.tensor(data_x_test_wholes, dtype=torch.float32).to(device)
    X_test_halves_tensor = torch.tensor(data_x_test_halves, dtype=torch.float32).to(device)


    with torch.no_grad():

        predictions_wholes = model(X_test_wholes_tensor)
        test_loss_wholes = criterion(predictions_wholes, Y_test_wholes).item()

        predictions_halves = model(X_test_halves_tensor)
        test_loss_halves = criterion(predictions_halves, Y_test_halves).item()



    

    criterion_name = criterion.__class__.__name__
    print(f"\n--- Model Evaluation ---")
    print(f"Wholes Test {criterion_name}: {test_loss_wholes:.4e}")
    print(f"Halves Test {criterion_name}: {test_loss_halves:.4e}")





    with open(LOGS_PATH, "a+") as logs_file:
        logs_file.write(f"Wholes Test {criterion_name}: {test_loss_wholes:.4e}\n")
        logs_file.write(f"Halves Test {criterion_name}: {test_loss_halves:.4e}\n")
    return  test_loss_wholes, test_loss_halves


Y = torch.stack([
    torch.tensor(normalized_data_dose,    dtype=torch.float32),
    torch.tensor(normalized_data_fluence_protons, dtype=torch.float32),
    torch.tensor(normalized_data_dlet_protons,     dtype=torch.float32),
], dim=1).to(device)
X_tensor = torch.tensor(normalized_x, dtype=torch.float32).to(device)
n_samples = n_samples = len(X_tensor)


weights_mask_train = np.repeat(weights_mask_train[:, np.newaxis, :], 3, axis=1)
weights_tensor_train = torch.tensor(weights_mask_train, device=device)

def weighted_mse_loss(pred, target, idx):
    eps = 1e-9
    sq_err_batch = (pred - target) / (target.abs() + eps)

    indexes = idx.detach().cpu().numpy() // seeds_per_energy
    weighted = (sq_err_batch**2) * weights_tensor_train[indexes]

    return weighted.mean()


weights_mask_test = np.repeat(weights_mask_test[:, np.newaxis, :], 3, axis=1)
weights_tensor_test = torch.tensor(weights_mask_test, device=device)

def test_weighted_mse(model):
    eps = 1e-9
    model.eval()

    normalized_data_dose_test_wholes, normalized_data_dose_test_halves = normalized_data_dose_test[::2], normalized_data_dose_test[1::2]
    normalized_data_fluence_protons_test_wholes, normalized_data_fluence_protons_test_halves = normalized_data_fluence_protons_test[::2], normalized_data_fluence_protons_test[1::2]
    normalized_data_dlet_protons_test_wholes, normalized_data_dlet_protons_test_halves = normalized_data_dlet_protons_test[::2], normalized_data_dlet_protons_test[1::2]
    data_x_test_wholes, data_x_test_halves = data_x_test[::2], data_x_test[1::2]



    Y_test_wholes = torch.stack(
        [
            torch.tensor(normalized_data_dose_test_wholes, dtype=torch.float32),
            torch.tensor(normalized_data_fluence_protons_test_wholes, dtype=torch.float32),
            torch.tensor(normalized_data_dlet_protons_test_wholes, dtype=torch.float32),
        ],
        dim=1,
    ).to(device)

    Y_test_halves = torch.stack(
        [
            torch.tensor(normalized_data_dose_test_halves, dtype=torch.float32),
            torch.tensor(normalized_data_fluence_protons_test_halves, dtype=torch.float32),
            torch.tensor(normalized_data_dlet_protons_test_halves, dtype=torch.float32),
        ],
        dim=1,
    ).to(device)

    X_test_wholes_tensor = torch.tensor(data_x_test_wholes, dtype=torch.float32).to(device)
    X_test_halves_tensor = torch.tensor(data_x_test_halves, dtype=torch.float32).to(device)

    weights_testor_test_wholes, weights_testor_test_halves = weights_tensor_test[::2], weights_tensor_test[1::2]
    with torch.no_grad():
        predictions_wholes = model(X_test_wholes_tensor)

        sq_err_batch = (predictions_wholes - Y_test_wholes) / (Y_test_wholes.abs() + eps)
        weighted_test_loss_wholes = (sq_err_batch**2) * weights_testor_test_wholes

        predictions_halves = model(X_test_halves_tensor)
        sq_err_batch = (predictions_halves - Y_test_halves) / (Y_test_halves.abs() + eps)
        weighted_test_loss_halves = (sq_err_batch**2) * weights_testor_test_halves

    return weighted_test_loss_wholes.mean(), weighted_test_loss_halves.mean(), 


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

batch_size = 128
total_epochs = 300

model     = Model().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=20, factor=0.5)
# criterion = nn.MSELoss()
# criterion = weighted_mse_loss

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
        loss = weighted_mse_loss(pred, y_batch, idx)
        # loss = criterion(pred, y_batch, idx)
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
        with open(LOGS_PATH,"a+") as logs_file:
            logs_file.write(
                f"Epoch {epoch:>4d} | Train loss: {train_loss:.8e}  | LR: {current_lr:.2e} | Time: {time.time()-start_training:.2f}\n"
            )

final_test_loss_wholes, final_test_loss_halves = test_weighted_mse(model)

CHECKPOINTS_DIR_NAME = "checkpoints"+slurm_job_id
def save_checkpoints():
    with open(Path(HOME,CHECKPOINTS_DIR_NAME, "best_losses_history"), "a") as best_losses_history:
        best_losses_history.write(f"{final_test_loss_wholes},{final_test_loss_halves}\n")
    with open(Path(HOME,CHECKPOINTS_DIR_NAME, "best_losses_history"), "r") as best_losses_history:
        losses_number = len(best_losses_history.readlines())-1
    torch.save(model.state_dict(), Path(HOME,CHECKPOINTS_DIR_NAME,f"best{str(losses_number)}.pth"))
    torch.save(model, Path(HOME,CHECKPOINTS_DIR_NAME,f"best_model{str(losses_number)}.pth"))
    with open(Path(HOME, CHECKPOINTS_DIR_NAME, "best_loss"), "w") as best_loss_file:
        best_loss_file.write(f"{final_test_loss_wholes},{final_test_loss_halves}")
        # os.makedirs('./checkpoints', exist_ok=True)
    # with open(Path(HOME, CHECKPOINTS_DIR_NAME, "best_code"+str(losses_number)), "w") as best_code_file:
    #     with open(Path(HOME,"TMP_DIR","train_model_initial_wieghted_loss.py"), "r") as current_code_file:
    #         best_code_file.write(current_code_file.read())

alpha = 0.75
if not Path(HOME, CHECKPOINTS_DIR_NAME).is_dir():
    Path(HOME, CHECKPOINTS_DIR_NAME).mkdir()
    with open(Path(HOME, CHECKPOINTS_DIR_NAME, "best_losses_history"),"w"):
        pass
    Path(HOME, CHECKPOINTS_DIR_NAME, "")
    save_checkpoints()
else:
    with open(Path(HOME, CHECKPOINTS_DIR_NAME, "best_loss"), "r") as best_loss_file:
        best_loss_tuple = best_loss_file.readline()
        best_loss_wholes, best_loss_halves = best_loss_tuple.split(",")
        best_loss_wholes, best_loss_halves = float(best_loss_wholes), float(best_loss_halves) 

        combined_best_loss = alpha * best_loss_wholes + (1 - alpha) * best_loss_halves
        combined_loss = alpha * final_test_loss_wholes + (1 - alpha) * final_test_loss_halves
        
    if final_test_loss_wholes < best_loss_wholes or final_test_loss_halves < best_loss_halves:
        save_checkpoints()
