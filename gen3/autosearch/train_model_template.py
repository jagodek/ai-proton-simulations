import os
import time
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import math
from pathlib import Path

{imports_definitions}
{additional_functions_definitions}

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

normalized_x = (data_x - x_min) / (x_max - x_min)
normalized_data_dose = data_dose / max_dose
normalized_data_fluence_protons = data_fluence_protons / max_fluence_protons
normalized_data_dlet_protons = data_dlet_protons / max_dlet_protons


test_data = np.load(Path(HOME, "test_data_g3batch10.npz"))
data_dose_test = test_data["data_dose_test"]
data_fluence_protons_test = test_data["data_fluence_protons_test"]
data_dlet_protons_test = test_data["data_dlet_protons_test"]
X_test = test_data["data_x_test"]

normalized_x_test = (X_test - x_min) / (x_max - x_min)
normalized_data_dose_test = data_dose_test / max_dose
normalized_data_fluence_protons_test = data_fluence_protons_test / max_fluence_protons
normalized_data_dlet_protons_test = data_dlet_protons_test / max_dlet_protons

with open(LOGS_PATH, "w+") as f:
    pass


def test_model(model, criterion, device):
    """
    Evaluates the trained model on the test dataset.
    """
    model.eval()

    Y_test = torch.stack(
        [
            torch.tensor(normalized_data_dose_test, dtype=torch.float32),
            torch.tensor(normalized_data_fluence_protons_test, dtype=torch.float32),
            torch.tensor(normalized_data_dlet_protons_test, dtype=torch.float32),
        ],
        dim=1,
    ).to(device)

    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)


    with torch.no_grad():
        predictions = model(X_test_tensor)
        test_loss = criterion(predictions, Y_test).item()
    criterion_name = criterion.__class__.__name__
    print(f"\n--- Model Evaluation ---")
    print(f"Total Test {criterion_name}: {test_loss:.4e}")
    with open(LOGS_PATH, "a+") as logs_file:
        logs_file.write(f"Total Test {criterion_name}: {test_loss:.4e}")
    return test_loss



Y = torch.stack(
    [
        torch.tensor(normalized_data_dose, dtype=torch.float32),
        torch.tensor(normalized_data_fluence_protons, dtype=torch.float32),
        torch.tensor(normalized_data_dlet_protons, dtype=torch.float32),
    ],
    dim=1,
).to(device)
X_tensor = torch.tensor(normalized_x, dtype=torch.float32).to(device)
n_samples = len(X_tensor)


{model_definition}


batch_size = {batch_size_definition}
total_epochs = {total_epochs_definition}

model = Model().to(device)
optimizer = {optimizer_definition}
scheduler = {scheduler_definition}
criterion = {criterion_definition}


model.train()
start_training = time.time()


{training_loop_definition}

final_test_loss = test_model(model, criterion, device, batch_size=batch_size)


def save_checkpoints():
    with open(Path(HOME, "checkpoints", "best_loss"), "w") as best_loss_file:
        best_loss_file.write(str(final_test_loss))
        # os.makedirs('./checkpoints', exist_ok=True)
    with open(Path(HOME, "checkpoints", "best_code"), "w") as best_code_file:
        with open(Path(HOME,"tmp","train_model_loop.py"), "r") as current_code_file:
            best_code_file.write(current_code_file.read())

    torch.save(model.state_dict(), Path(HOME,"checkpoints","best.pth"))
    torch.save(model, Path(HOME,"checkpoints","best_model.pth"))



if not Path(HOME, "checkpoints", "best_loss").is_file():
    save_checkpoints()
else:
    with open(Path(HOME, "checkpoints", "best_loss"), "r") as best_loss_file:
        best_loss = float(best_loss_file.readline())

    if final_test_loss < best_loss:
        save_checkpoints()
