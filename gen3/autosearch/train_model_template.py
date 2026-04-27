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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
training_data = np.load("training_data_g3batch7.npz")
data_dose = training_data["data_dose"]
data_fluence_protons = training_data["data_fluence_protons"]
data_dlet_protons = training_data["data_dlet_protons"]
X = training_data["data_x"]
x_min, x_max = np.min(X), np.max(X)
max_dose = np.max(data_dose)
max_fluence_protons = np.max(data_fluence_protons)
max_dlet_protons = np.max(data_dlet_protons)

normalized_x = (X - x_min) / (x_max-x_min)
normalized_data_dose = data_dose/max_dose
normalized_data_fluence_protons = data_fluence_protons/max_fluence_protons
normalized_data_dlet_protons = data_dlet_protons/max_dlet_protons


test_data = np.load("test_data_g3batch8.npz")
data_dose_test = test_data["data_dose_test"]
data_fluence_protons_test = test_data["data_fluence_protons_test"]
data_dlet_protons_test = test_data["data_dlet_protons_test"]
X_test = test_data["data_x_test"]

normalized_x_test = (X_test - x_min) / (x_max-x_min)
normalized_data_dose_test = data_dose_test/max_dose
normalized_data_fluence_protons_test = data_fluence_protons_test/max_fluence_protons
normalized_data_dlet_protons_test = data_dlet_protons_test/max_dlet_protons


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
    
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    n_test = len(X_test_tensor)
    
    test_loss = 0.0
    
    # 3. Disable gradient computation to save memory and speed up inference
    with torch.no_grad():
        for i in range(0, n_test, batch_size):
            x_batch = X_test_tensor[i:i+batch_size]
            y_batch = Y_test[i:i+batch_size]
            
            # Forward pass
            pred = model(x_batch)
            loss = criterion(pred, y_batch)
            test_loss += loss.item()
            
    # Calculate the average test loss across all batches
    num_batches = math.ceil(n_test / batch_size)
    avg_test_loss = test_loss / num_batches
    
    print(f"\n--- Model Evaluation ---")
    print(f"Final Test MSE Loss: {avg_test_loss:.4e}")
    with open("logs","a") as logs_file:
            logs_file.write(
                f"Final Test MSE Loss: {avg_test_loss:.4e}"
            )
    return avg_test_loss

seeds_per_energy = 0
for i,x in enumerate(X):
    if x == X[0]:
        seeds_per_energy += 1
    else:
        break
print(seeds_per_energy)




Y = torch.stack([
    torch.tensor(normalized_data_dose,    dtype=torch.float32),
    torch.tensor(normalized_data_fluence_protons, dtype=torch.float32),
    torch.tensor(normalized_data_dlet_protons,     dtype=torch.float32),
], dim=1).to(device)
X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
n = len(X_tensor)


{model_definition}


model     = Model().to(device)
optimizer = {optimizer_definition}
scheduler = {scheduler_definition}
criterion = {criterion_definition}

best_val_loss = float("inf")

model.train()
n_samples = X.shape[0]
batch_size = {batch_size_definition}
total_epochs = {total_epochs_definition}
start_training = time.time()
with open("logs","w") as logs_file:
    pass
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

    



    if epoch % 50 == 0:
        print(
            f"Epoch {epoch:>4d} | ",
            f"Train loss: {train_loss:.4e} | ",
            f"LR: {current_lr:.2e}",
            f"Time: {time.time()-start_training:.2f}"
        )
        with open("logs","a") as logs_file:
            logs_file.write(
                f"Epoch {epoch:>4d} | Train loss: {train_loss:.8e}  | LR: {current_lr:.2e} | Time: {time.time()-start_training:.2f}\n"
            )

final_test_loss = test_model(model, criterion, device, batch_size=128)

with open("best_loss", "r") as best_loss_file:
    best_loss = float(best_loss_file.readline())

if final_test_loss < best_loss:
    with open("best_loss", "w") as best_loss_file:
        best_loss_file.write(final_test_loss)
        # os.makedirs('./checkpoints', exist_ok=True)
        torch.save(model.state_dict(), './checkpoints/best.pth')
        torch.save(model, './checkpoints/best_model.pth')