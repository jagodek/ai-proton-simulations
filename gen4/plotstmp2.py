# %%
import os
import json
import random
import shutil
import re
import math
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import multiprocessing as mp
from matplotlib import pyplot as plt
from pathlib import Path
# load_dotenv(find_dotenv(usecwd=True))

HOME = "/home/michal/slrm/gen4"
if os.getenv("PLG_GROUPS_STORAGE"):
    HOME = "/net/people/plgrid/plgmichalgodek/workspace/ai-proton-simulations/gen4/"
os.chdir(HOME)

# %%

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
training_data = np.load(Path(HOME,"training_data_g4_batch.npz"))
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
# overall max (kept for compatibility)

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

test_data = np.load(Path(HOME,"test_data_g3batch10.npz"))
data_z_dose_test = test_data["data_z_dose_test"]
data_r_dose_test = test_data["data_r_dose_test"]
data_z_fluence_protons_test = test_data["data_z_fluence_protons_test"]
data_r_fluence_protons_test = test_data["data_r_fluence_protons_test"]
data_z_dlet_protons_test = test_data["data_z_dlet_protons_test"]
data_r_dlet_protons_test = test_data["data_r_dlet_protons_test"]
data_x_test = test_data["data_x_test"]

normalized_x_test = (data_x_test - x_min) / (x_max-x_min)
normalized_data_z_dose_test = data_z_dose_test / max_z_dose
normalized_data_r_dose_test = data_r_dose_test / max_r_dose
normalized_data_z_fluence_protons_test = data_z_fluence_protons_test / max_z_fluence_protons
normalized_data_r_fluence_protons_test = data_r_fluence_protons_test / max_r_fluence_protons
normalized_data_z_dlet_protons_test = data_z_dlet_protons_test / max_z_dlet_protons
normalized_data_r_dlet_protons_test = data_r_dlet_protons_test / max_r_dlet_protons

seeds_per_energy = 0
for i,x in enumerate(data_x):
    if float(x) == data_x[0]:
        seeds_per_energy += 1
    else:
        break

print(seeds_per_energy)

# %%
import plotly.io as pio
pio.renderers.default = "notebook_connected" 

# %%
plt.figure(figsize=(10,10))
for i in data_z_dose[::1]: 
    plt.plot(i)
plt.show()

plt.figure(figsize=(10,10))
for i in data_z_fluence_protons[::1]: 
    plt.plot(i)
plt.show()

plt.figure(figsize=(10,10))
for i in data_z_dlet_protons[::1]: 
    plt.plot(i)
plt.show()

plt.figure(figsize=(10,10))
for i in data_r_dose[::1]: 
    plt.plot(i)
plt.show()

plt.figure(figsize=(10,10))
for i in data_r_fluence_protons[::1]: 
    plt.plot(i)
plt.show()

plt.figure(figsize=(10,10))
for i in data_r_dlet_protons[::1]: 
    plt.plot(i)
plt.show()

# %%

plt.figure(figsize=(10,10))
for i in data_z_dose_test[::1]: 
    plt.plot(i)
plt.show()

plt.figure(figsize=(10,10))
for i in data_z_fluence_protons_test[::1]: 
    plt.plot(i)
plt.show()

plt.figure(figsize=(10,10))
for i in data_z_dlet_protons_test[::1]: 
    plt.plot(i)
plt.show()

plt.figure(figsize=(10,10))
for i in data_r_dose_test[::1]: 
    plt.plot(i)
plt.show()

plt.figure(figsize=(10,10))
for i in data_r_fluence_protons_test[::1]: 
    plt.plot(i)
plt.show()

plt.figure(figsize=(10,10))
for i in data_r_dlet_protons_test[::1]: 
    plt.plot(i)
plt.show()

# %%
# fig, ax = plt.subplots(figsize=(10,10))
# # plt.figure(figsize=(10,10))

# ax.set_yscale('log', base=10)
# for i in data_z_dose[::1]: 
#     ax.plot(i)
# fig.show()


# plt.figure(figsize=(10,10))
# for i in data_z_fluence_protons[::1]: 
#     plt.plot(i)
# plt.show()

# plt.figure(figsize=(10,10))
# for i in data_z_dlet_protons[::1]: 
#     plt.plot(i)
# plt.show()

# %%
# import plotly.graph_objects as go
# fig = go.Figure()
# fig.update_layout(
#     width=900,   # px, default is 700
#     height=700,   # px, default is 450
#     autosize=False,  # must be False to fix the size
# )
# x = list(range(400))

# for i in range(1,31):
#     fig.add_trace(go.Scatter(x=x, y=data_z_dose[-i,:], mode="lines", showlegend=False, opacity = 1, line={"color": "red"}))
#     # print(data_x[-i])
# fig.add_trace(go.Scatter(x=x, y=data_z_dose_test[-1,:], mode="lines", showlegend=False, opacity = 1, line={"color": "blue"}))
# fig.show()

# %% [markdown]
# # Load model

# %%

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

# %%
model = torch.load( HOME+'/checkpoints/basic_model.pth', weights_only=False,  map_location=device)#edit before run
# state_dict = torch.load( './checkpoints/gen3_batch2.pth',  map_location=device)
# model  = Model()
# model.load_state_dict(state_dict)
# model = model.to(device)
model = model.to(device)
model.eval()

# %% [markdown]
# # Make predictions

# %%
# print(normalized_x)

# %%
model.eval()
X_norm_unique = np.unique(normalized_x).reshape(-1, 1)

with torch.no_grad():
    predictions = model(torch.from_numpy(X_norm_unique.astype(np.float32)).to(device))

# %%
for key, value in predictions.items():
    print(key)
    for distribution in value:
        plt.plot(distribution.cpu().numpy())
        plt.title(key)
    plt.show()

# %%
# print(X_norm_unique)

# %%
# X2 = X + 0.5
# # apply same min-max scaling to modified energies
# if X_max == X_min:
#     X2_norm = np.zeros_like(X2)
# else:
#     X2_norm = (X2 - X_min) / (X_max - X_min)
# X2_norm_unique = np.unique(X2_norm).reshape(-1, 1)


# with torch.no_grad():
#     predictions2 = model(torch.from_numpy(X2_norm_unique.astype(np.float32)).to(device)).cpu().numpy()

# %%
predictions.shape

# %%
plt.figure(figsize=(15,15))
for i in predictions['dose_z']:
    plt.plot(i)

# %%
criterion = nn.HuberLoss(delta=0.08)

# %%
# def test_model(model, criterion, device, batch_size=1):
#     """
#     Evaluates the trained model on the test dataset.
#     """
#     # 1. Set the model to evaluation mode
#     model.eval()
    
#     # 2. Prepare test tensors and move them to the correct device
#     Y_test = torch.stack([
#         torch.tensor(normalized_z_data_dose_test, dtype=torch.float32),
#         torch.tensor(normalized_data_z_fluence_protons_test, dtype=torch.float32),
#         torch.tensor(normalized_data_z_dlet_protons_test, dtype=torch.float32),
#     ], dim=1).to(device)
    
#     X_test_tensor = torch.tensor(data_x_test, dtype=torch.float32).to(device)
#     n_test = len(X_test_tensor)
    
#     total_test_loss = 0.0
    
#     # 3. Disable gradient computation to save memory and speed up inference
#     with torch.no_grad():
#         for i in range(0, n_test, batch_size):
#             x_batch = X_test_tensor[i:i+batch_size]
#             y_batch = Y_test[i:i+batch_size]
            
#             # Forward pass
#             pred = model(x_batch)
#             loss = criterion(pred, y_batch)
#             total_test_loss += loss.item()
            
#     # Calculate the average test loss across all batches
#     # total_test_loss = math.ceil(n_test / batch_size)
#     # avg_test_loss = test_loss / num_batches
    
#     return total_test_loss

# test_model(model, criterion, device)

# %% [markdown]
# ## Compare against training data

# %% [markdown]
# ## Plot predictions and true data

# %% [markdown]
# ### Dose z

# %%
import plotly.graph_objects as go
def prep_fig(title):
    fig = go.Figure()
    fig.update_layout(
        width=1200,   # px, default is 700
        height=1000,   # px, default is 450
        autosize=False,  # must be False to fix the size
        title = title,
    )
    return fig
x = list(range(400))

colors = ["red", "blue", "green", "orange"]
col_len = len(colors)

every_nth = 20
def plot_true_pred(true, pred, multiplier, quantity):

    metric = (pred-true)*multiplier

    fig = prep_fig(f"{quantity}:" + "for all energies" if every_nth == 1 else f"for every {every_nth}(th) energy" )
    for i, (tr,pr) in enumerate(zip(true[::every_nth],pred[::every_nth])):
        fig.add_trace(go.Scatter(x=x, y=pr*multiplier, mode="lines", showlegend=False, opacity = 1, line={"color": colors[i%col_len]}))
        fig.add_trace(go.Scatter(x=x, y=tr*multiplier, mode="lines", showlegend=False, opacity = 0.4, line={"color": colors[i%col_len]}))

    fig.show()
    
    fig = prep_fig(f"{quantity}: for lowest energy {data_x[0]} MeV")
    fig.add_trace(go.Scatter(x=x, y=pred[0]*multiplier, mode="lines", showlegend=False, opacity = 1, line={"color": colors[i%col_len]}))
    fig.add_trace(go.Scatter(x=x, y=true[0]*multiplier, mode="lines", showlegend=False, opacity = 0.4, line={"color": colors[i%col_len]}))


    fig.show()
    fig = prep_fig(f"{quantity}: for highest energy {data_x[-1]} MeV")
    fig.add_trace(go.Scatter(x=x, y=pred[-1]*multiplier, mode="lines", showlegend=False, opacity = 1, line={"color": colors[i%col_len]}))
    fig.add_trace(go.Scatter(x=x, y=true[-1]*multiplier, mode="lines", showlegend=False, opacity = 0.4, line={"color": colors[i%col_len]}))

    fig.show()    

plot_true_pred(normalized_data_z_dose[::seeds_per_energy], predictions['dose_z'], max_z_dose, "Dose z")


# %% [markdown]
# ### Fluence z

# %%
plot_true_pred(normalized_data_z_fluence_protons[::seeds_per_energy], predictions['fluence_z'], max_z_fluence_protons, "Fluence z")

# %% [markdown]
# ### dLET z

# %%
plot_true_pred(normalized_data_z_dlet_protons[::seeds_per_energy], predictions['dlet_z'], max_z_dlet_protons, "dLet z")

# %% [markdown]
# ### Dose r

plot_true_pred(normalized_data_r_dose[::seeds_per_energy], predictions['dose_r'], max_r_dose, "Dose r")

# %% [markdown]
# ### Fluence r

# %%
plot_true_pred(normalized_data_r_fluence_protons[::seeds_per_energy], predictions['fluence_r'], max_r_fluence_protons, "Fluence r")

# %% [markdown]
# ### dLET r

# %%
plot_true_pred(normalized_data_r_dlet_protons[::seeds_per_energy], predictions['dlet_r'], max_r_dlet_protons, "dLet r")


# %% [markdown]
# ## Logarythmic scale

# %% [markdown]
# ### Dose

# %%
import plotly.graph_objects as go
def prep_fig(title):
    fig = go.Figure()
    fig.update_layout(
        width=1200,   # px, default is 700
        height=1000,   # px, default is 450
        autosize=False,  # must be False to fix the size
        title = title,
    )
    return fig
x = list(range(400))

colors = ["red", "blue", "green", "orange"]
col_len = len(colors)

every_nth = 10
def plot_true_pred_log(true, pred, multiplier, quantity):

    metric = (pred-true)*multiplier

    fig = prep_fig(f"{quantity}:  for all energies")
    for i, (tr,pr) in enumerate(zip(true,pred)):
        fig.add_trace(go.Scatter(x=x, y=pr*multiplier, mode="lines", showlegend=False, opacity = 1, line={"color": colors[i%col_len]}))
        fig.add_trace(go.Scatter(x=x, y=tr*multiplier, mode="lines", showlegend=False, opacity = 0.4, line={"color": colors[i%col_len]}))
    fig.update_yaxes(type="log")
    fig.show()
    
    fig = prep_fig(f"{quantity}: for lowest energy {data_x[0]} MeV")
    fig.add_trace(go.Scatter(x=x, y=pred[0]*multiplier, mode="lines", showlegend=False, opacity = 1, line={"color": colors[i%col_len]}))
    fig.add_trace(go.Scatter(x=x, y=true[0]*multiplier, mode="lines", showlegend=False, opacity = 0.4, line={"color": colors[i%col_len]}))
    fig.update_yaxes(type="log")
    fig.show()
    
    fig = prep_fig(f"{quantity}: for highest energy {data_x[-1]} MeV")
    fig.add_trace(go.Scatter(x=x, y=pred[-1]*multiplier, mode="lines", showlegend=False, opacity = 1, line={"color": colors[i%col_len]}))
    fig.add_trace(go.Scatter(x=x, y=true[-1]*multiplier, mode="lines", showlegend=False, opacity = 0.4, line={"color": colors[i%col_len]}))
    fig.update_yaxes(type="log")
    fig.show()    

plot_true_pred_log(normalized_data_z_dose[::seeds_per_energy], predictions['dose_z'], max_z_dose, "Dose z")


# %% [markdown]
# ### Fluence z

# %%
plot_true_pred_log(normalized_data_z_fluence_protons[::seeds_per_energy],  predictions['fluence_z'], max_z_fluence_protons, "Fluence z")

# %% [markdown]
# ### dLet z

# %%
plot_true_pred_log(normalized_data_z_dlet_protons[::seeds_per_energy],  predictions['dlet_z'], max_z_dlet_protons, "dLet z")

# %% [markdown]
# ## difference

# %% [markdown]
# ### Dose r

# %%
plot_true_pred_log(normalized_data_r_dose[::seeds_per_energy], predictions['dose_r'], max_r_dose, "Dose r")
# %% [markdown]
# ### Fluence r

# %%
plot_true_pred_log(normalized_data_r_fluence_protons[::seeds_per_energy],  predictions['fluence_r'], max_r_fluence_protons, "Fluence r")

# %% [markdown]
# ### dLet r

# %%
plot_true_pred_log(normalized_data_r_dlet_protons[::seeds_per_energy],  predictions['dlet_r'], max_r_dlet_protons, "dLet r")



# %% [markdown]
# ### dose difference 

# %%


# %%
def prep_fig(title):
    fig = go.Figure()
    fig.update_layout(
        width=1200,   # px, default is 700
        height=1000,   # px, default is 450
        autosize=False,  # must be False to fix the size
        title = title
    )
    return fig
x = list(range(400))

colors = ["red", "blue", "green", "orange"]
col_len = len(colors)

every_nth = 10
def plot_diff(true, pred, multiplier, quantity):

    metric = (pred-true)*multiplier

    fig = prep_fig(f"{quantity}: Difference for all energies")
    for i,serie in enumerate(metric):
        fig.add_trace(go.Scatter(x=x, y=serie, mode="lines", showlegend=False, opacity = 1, line={"color": colors[i%col_len]}))

    fig.show()
    
    fig = prep_fig(f"{quantity}: Difference for lowest energy")
    fig.add_trace(go.Scatter(x=x, y=metric[0], mode="lines", showlegend=False, opacity = 1, line={"color": colors[i%col_len]}))


    fig.show()
    fig = prep_fig(f"{quantity}: Difference for highest energy")
    fig.add_trace(go.Scatter(x=x, y=metric[-1], mode="lines", showlegend=False, opacity = 1, line={"color": colors[i%col_len]}))


    fig.show()    

plot_diff(normalized_data_z_dose[::seeds_per_energy], predictions['dose_z'], max_z_dose, "Dose z")

# %% [markdown]
# ### Fluence z difference 

# %%
plot_diff(normalized_data_z_fluence_protons[::seeds_per_energy], predictions['fluence_z'], max_z_dlet_protons, "Fluence z")

# %% [markdown]
# ### Dlet z difference 

# %%
plot_diff(normalized_data_z_dlet_protons[::seeds_per_energy], predictions['dlet_z'], max_z_dlet_protons, "DLet z")

# %% [markdown]
# ### Dose r difference 

# %%
plot_diff(normalized_data_r_fluence_protons[::seeds_per_energy], predictions['fluence_r'], max_r_dlet_protons, "Fluence r")

# %% [markdown]
# ### Fluence r difference 

# %%
plot_diff(normalized_data_r_fluence_protons[::seeds_per_energy], predictions['fluence_r'], max_r_dlet_protons, "Fluence r")

# %% [markdown]
# ### DLet r difference 

# %%
plot_diff(normalized_data_r_dlet_protons[::seeds_per_energy], predictions['dlet_r'], max_r_dlet_protons, "Dlet r")


# %% [markdown]
# ## Percent diff

# %% [markdown]
# ### Dose z percent difference

# %%
def print_percent_diff(true, pred, quantity):

    metric = (true/pred-1)*100

    fig = prep_fig(f"{quantity}: Percent difference for all energies")
    for i,serie in enumerate(metric):
        fig.add_trace(go.Scatter(x=x, y=serie, mode="lines", showlegend=False, opacity = 1, line={"color": colors[i%col_len]}))

    fig.show()
    
    fig = prep_fig(f"{quantity}: Percent difference for lowest energy")
    fig.add_trace(go.Scatter(x=x, y=metric[0], mode="lines", showlegend=False, opacity = 1, line={"color": colors[i%col_len]}))


    fig.show()
    fig = prep_fig(f"{quantity}: Percent difference for highest energy")
    fig.add_trace(go.Scatter(x=x, y=metric[-1], mode="lines", showlegend=False, opacity = 1, line={"color": colors[i%col_len]}))


    fig.show()    

print_percent_diff(normalized_data_z_dose[::seeds_per_energy], predictions['dose_z'], "Dose z")

# %% [markdown]
# ### Fluence z percent difference

# %%
print_percent_diff(normalized_data_z_fluence_protons[::seeds_per_energy], predictions['fluence_z'], "Fluence z")

# %% [markdown]
# ### Dlet z percent difference

# %%
print_percent_diff(normalized_data_z_dlet_protons[::seeds_per_energy], predictions['dlet_z'], "dLet z")

# %% [markdown]
# ### Dose r percent difference

# %%
print_percent_diff(normalized_data_r_dose[::seeds_per_energy], predictions['dose_r'], "Dose r")

# %% [markdown]
# ### Fluence r percent difference

# %%
print_percent_diff(normalized_data_r_fluence_protons[::seeds_per_energy], predictions['fluence_r'], "Fluence r")

# %% [markdown]
# ### Dlet r percent difference

# %%
print_percent_diff(normalized_data_r_dlet_protons[::seeds_per_energy], predictions['dlet_r'], "DLet r")

# %% [markdown]
# # Compare against test data

# %%
model.eval()

with torch.no_grad():
    test_predictions = model(torch.from_numpy(normalized_x_test.astype(np.float32)).to(device)).cpu().numpy()
# test_predictions = test_predictions[1::2]


test_predictions_wholes = test_predictions[::2]
test_predictions_halves = test_predictions[1::2]

normalized_test_x_wholes, normalized_test_x_halves = normalized_x_test[::2], normalized_x_test[1::2]


normalized_data_z_dose_test_wholes, normalized_data_z_dose_test_halves = normalized_data_z_dose_test[::2], normalized_data_z_dose_test[1::2] 
normalized_data_z_fluence_protons_test_wholes, normalized_data_z_fluence_protons_test_halves = normalized_data_z_fluence_protons_test[::2], normalized_data_z_fluence_protons_test[1::2] 
normalized_data_z_dlet_protons_test_wholes, normalized_data_z_dlet_protons_test_halves = normalized_data_z_dlet_protons_test[::2], normalized_data_z_dlet_protons_test[1::2] 

normalized_data_r_dose_test_wholes, normalized_data_r_dose_test_halves = normalized_data_r_dose_test[::2], normalized_data_r_dose_test[1::2] 
normalized_data_r_fluence_protons_test_wholes, normalized_data_r_fluence_protons_test_halves = normalized_data_r_fluence_protons_test[::2], normalized_data_r_fluence_protons_test[1::2] 
normalized_data_r_dlet_protons_test_wholes, normalized_data_r_dlet_protons_test_halves = normalized_data_r_dlet_protons_test[::2], normalized_data_r_dlet_protons_test[1::2] 
# %%
print(test_predictions.shape)

# %%
plt.figure(figsize=(10,10))
plt.plot(predictions[0,0])
plt.plot(predictions[1,0])
plt.plot(test_predictions[1,0])

# %%
test_predictions.shape
# %% [markdown]
# ### Dose z


# %%
import plotly.graph_objects as go
def prep_fig(title):
    fig = go.Figure()
    fig.update_layout(
        width=1200,   # px, default is 700
        height=1000,   # px, default is 450
        autosize=False,  # must be False to fix the size
        title = title,
    )
    return fig
x = list(range(400))


colors = ["red", "blue", "green", "orange"]
col_len = len(colors)


every_nth = 10
def plot_true_pred(true, pred, multiplier, quantity):


    metric = (pred-true)*multiplier


    fig = prep_fig(f"{quantity}:  for all energies")
    for i, (tr,pr) in enumerate(zip(true[::20],pred[::20])):
        fig.add_trace(go.Scatter(x=x, y=pr*multiplier, mode="lines", showlegend=False, opacity = 1, line={"color": colors[i%col_len]}))
        fig.add_trace(go.Scatter(x=x, y=tr*multiplier, mode="lines", showlegend=False, opacity = 0.4, line={"color": colors[i%col_len]}))


    fig.show()
    
    fig = prep_fig(f"{quantity}: for lowest energy {data_x_test[1]} MeV")
    fig.add_trace(go.Scatter(x=x, y=pred[0]*multiplier, mode="lines", showlegend=False, opacity = 1, line={"color": colors[i%col_len]}))
    fig.add_trace(go.Scatter(x=x, y=true[0]*multiplier, mode="lines", showlegend=False, opacity = 0.4, line={"color": colors[i%col_len]}))



    fig.show()
    fig = prep_fig(f"{quantity}: for highest energy {data_x_test[-1]} MeV")
    fig.add_trace(go.Scatter(x=x, y=pred[-1]*multiplier, mode="lines", showlegend=False, opacity = 1, line={"color": colors[i%col_len]}))
    fig.add_trace(go.Scatter(x=x, y=true[-1]*multiplier, mode="lines", showlegend=False, opacity = 0.4, line={"color": colors[i%col_len]}))



    fig.show()    


plot_true_pred(normalized_data_z_dose_test_wholes, test_predictions_wholes['dose_z'], max_z_dose, "Dose z")



# %% [markdown]
# ### Fluence z


# %%
plot_true_pred(normalized_data_z_fluence_protons_test_wholes[::], test_predictions_wholes['fluence_z'], max_z_fluence_protons, "Fluence z")


# %% [markdown]
# ### dLET z


# %%
plot_true_pred(normalized_data_z_dlet_protons_test_wholes[::], test_predictions_wholes['dlet_z'], max_z_dlet_protons, "dLet z")



# %% [markdown]
# ## Logarythmic scale


# %% [markdown]
# ### Dose z


# %%
import plotly.graph_objects as go
def prep_fig(title):
    fig = go.Figure()
    fig.update_layout(
        width=1200,   # px, default is 700
        height=1000,   # px, default is 450
        autosize=False,  # must be False to fix the size
        title = title,
    )
    return fig
x = list(range(400))


colors = ["red", "blue", "green", "orange"]
col_len = len(colors)


every_nth = 10
def plot_true_pred_log(true, pred, multiplier, quantity):


    metric = (pred-true)*multiplier


    fig = prep_fig(f"{quantity}:  for all energies")
    for i, (tr,pr) in enumerate(zip(true[::20],pred[::20])):
        fig.add_trace(go.Scatter(x=x, y=pr*multiplier, mode="lines", showlegend=False, opacity = 1, line={"color": colors[i%col_len]}))
        fig.add_trace(go.Scatter(x=x, y=tr*multiplier, mode="lines", showlegend=False, opacity = 0.4, line={"color": colors[i%col_len]}))
    fig.update_yaxes(type="log")
    fig.show()
    
    fig = prep_fig(f"{quantity}: for lowest energy {data_x[0]} MeV")
    fig.add_trace(go.Scatter(x=x, y=pred[0]*multiplier, mode="lines", showlegend=False, opacity = 1, line={"color": colors[i%col_len]}))
    fig.add_trace(go.Scatter(x=x, y=true[0]*multiplier, mode="lines", showlegend=False, opacity = 0.4, line={"color": colors[i%col_len]}))
    fig.update_yaxes(type="log")
    fig.show()
    
    fig = prep_fig(f"{quantity}: for highest energy {data_x[-1]} MeV")
    fig.add_trace(go.Scatter(x=x, y=pred[-1]*multiplier, mode="lines", showlegend=False, opacity = 1, line={"color": colors[i%col_len]}))
    fig.add_trace(go.Scatter(x=x, y=true[-1]*multiplier, mode="lines", showlegend=False, opacity = 0.4, line={"color": colors[i%col_len]}))
    fig.update_yaxes(type="log")
    fig.show()    


plot_true_pred_log(normalized_data_z_dose_test_wholes[::], test_predictions_wholes['dose_z'], max_z_dose, "Dose z")



# %% [markdown]
# ### Fluence z


# %%
plot_true_pred_log(normalized_data_z_fluence_protons_test_wholes[::],  test_predictions_wholes['fluence_z'], max_z_fluence_protons, "Fluence z")


# %% [markdown]
# ### dLet z


# %%
plot_true_pred_log(normalized_data_z_dlet_protons_test_wholes[::],  test_predictions_wholes['dlet_z'], max_z_dlet_protons, "dLet z")


# %% [markdown]
# ## difference


# %% [markdown]
# ### dose z difference 


# %%
def prep_fig(title):
    fig = go.Figure()
    fig.update_layout(
        width=1200,   # px, default is 700
        height=1000,   # px, default is 450
        autosize=False,  # must be False to fix the size
        title = title,
    )
    return fig
x = list(range(400))


colors = ["red", "blue", "green", "orange"]
col_len = len(colors)


every_nth = 10
def plot_diff(true, pred, multiplier, quantity):


    metric = (pred-true)*multiplier


    fig = prep_fig(f"{quantity}: Difference for all energies")
    for i,serie in enumerate(metric):
        fig.add_trace(go.Scatter(x=x, y=serie, mode="lines", showlegend=False, opacity = 1, line={"color": colors[i%col_len]}))


    fig.show()
    
    fig = prep_fig(f"{quantity}: Difference for lowest energy")
    fig.add_trace(go.Scatter(x=x, y=metric[0], mode="lines", showlegend=False, opacity = 1, line={"color": colors[i%col_len]}))



    fig.show()
    fig = prep_fig(f"{quantity}: Difference for highest energy")
    fig.add_trace(go.Scatter(x=x, y=metric[-1], mode="lines", showlegend=False, opacity = 1, line={"color": colors[i%col_len]}))



    fig.show()    


plot_diff(normalized_data_z_dose_test_wholes[::], test_predictions_wholes['dose_z'], max_z_dose, "Dose z")


# %% [markdown]
# ### fluence z difference 


# %%
plot_diff(normalized_data_z_fluence_protons_test_wholes[::], test_predictions_wholes['fluence_z'], max_z_dlet_protons, "Fluence z")


# %% [markdown]
# ### dlet z difference 


# %%
plot_diff(normalized_data_z_dlet_protons_test_wholes[::], test_predictions_wholes['dlet_z'], max_z_dlet_protons, "dlet z")


# %% [markdown]
# ## Percent diff


# %% [markdown]
# ### Dose z percent difference


# %%
def print_percent_diff(true, pred, quantity):


    metric = (true/pred-1)*100


    fig = prep_fig(f"{quantity}: Percent difference for all energies")
    for i,serie in enumerate(metric[::]):
        fig.add_trace(go.Scatter(x=x, y=serie, mode="lines", showlegend=False, opacity = 1, line={"color": colors[i%col_len]}))


    fig.show()
    
    fig = prep_fig(f"{quantity}: Percent difference for lowest energy")
    fig.add_trace(go.Scatter(x=x, y=metric[0], mode="lines", showlegend=False, opacity = 1, line={"color": colors[i%col_len]}))



    fig.show()
    fig = prep_fig(f"{quantity}: Percent difference for highest energy")
    fig.add_trace(go.Scatter(x=x, y=metric[-1], mode="lines", showlegend=False, opacity = 1, line={"color": colors[i%col_len]}))



    fig.show()    


print_percent_diff(normalized_data_z_dose_test_wholes[::], test_predictions_wholes['dose_z'], "Dose z")


# %% [markdown]
# ### Fluence z percent difference


# %%
print_percent_diff(normalized_data_z_fluence_protons_test_wholes[::], test_predictions_wholes['fluence_z'], "Fluence z")


# %% [markdown]
# ### Dlet z percent difference


# %%
print_percent_diff(normalized_data_z_dlet_protons_test_wholes[::], test_predictions_wholes['dlet_z'], "dLet z")


# %%



# %% [markdown]
# # Compare against test halves


# %% [markdown]
# # Compare against test halves


# %%
plt.figure(figsize=(10,10))
plt.plot(predictions[0,0], color="blue")
plt.plot(predictions[1,0], color="red")
plt.plot(test_predictions[1,0], color="yellow")


# %%
test_predictions.shape


# %% [markdown]
# ### Dose z


# %%
import plotly.graph_objects as go
def prep_fig(title):
    fig = go.Figure()
    fig.update_layout(
        width=1200,   # px, default is 700
        height=1000,   # px, default is 450
        autosize=False,  # must be False to fix the size
        title = title,
    )
    return fig
x = list(range(400))


colors = ["red", "blue", "green", "orange"]
col_len = len(colors)


every_nth = 10
def plot_true_pred(true, pred, multiplier, quantity):


    metric = (pred-true)*multiplier


    fig = prep_fig(f"{quantity}:  for all energies")
    for i, (tr,pr) in enumerate(zip(true[::20],pred[::20])):
        fig.add_trace(go.Scatter(x=x, y=pr*multiplier, mode="lines", showlegend=False, opacity = 1, line={"color": colors[i%col_len]}))
        fig.add_trace(go.Scatter(x=x, y=tr*multiplier, mode="lines", showlegend=False, opacity = 0.4, line={"color": colors[i%col_len]}))


    fig.show()
    
    fig = prep_fig(f"{quantity}: for lowest energy {data_x_test[1]} MeV")
    fig.add_trace(go.Scatter(x=x, y=pred[0]*multiplier, mode="lines", showlegend=False, opacity = 1, line={"color": colors[i%col_len]}))
    fig.add_trace(go.Scatter(x=x, y=true[0]*multiplier, mode="lines", showlegend=False, opacity = 0.4, line={"color": colors[i%col_len]}))



    fig.show()
    fig = prep_fig(f"{quantity}: for highest energy {data_x_test[-1]} MeV")
    fig.add_trace(go.Scatter(x=x, y=pred[-1]*multiplier, mode="lines", showlegend=False, opacity = 1, line={"color": colors[i%col_len]}))
    fig.add_trace(go.Scatter(x=x, y=true[-1]*multiplier, mode="lines", showlegend=False, opacity = 0.4, line={"color": colors[i%col_len]}))



    fig.show()    


plot_true_pred(normalized_data_z_dose_test_halves, test_predictions_halves['dose_z'], max_z_dose, "Dose z")



# %% [markdown]
# ### Fluence z


# %%
plot_true_pred(normalized_data_z_fluence_protons_test_halves[::], test_predictions_halves['fluence_z'], max_z_fluence_protons, "Fluence z")


# %% [markdown]
# ### dLET z


# %%
plot_true_pred(normalized_data_z_dlet_protons_test_halves[::], test_predictions_halves['dlet_z'], max_z_dlet_protons, "dLet z")



# %% [markdown]
# ## Logarythmic scale


# %% [markdown]
# ### Dose z


# %%
import plotly.graph_objects as go
def prep_fig(title):
    fig = go.Figure()
    fig.update_layout(
        width=1200,   # px, default is 700
        height=1000,   # px, default is 450
        autosize=False,  # must be False to fix the size
        title = title,
    )
    return fig
x = list(range(400))


colors = ["red", "blue", "green", "orange"]
col_len = len(colors)


every_nth = 10
def plot_true_pred_log(true, pred, multiplier, quantity):


    metric = (pred-true)*multiplier


    fig = prep_fig(f"{quantity}:  for all energies")
    for i, (tr,pr) in enumerate(zip(true[::20],pred[::20])):
        fig.add_trace(go.Scatter(x=x, y=pr*multiplier, mode="lines", showlegend=False, opacity = 1, line={"color": colors[i%col_len]}))
        fig.add_trace(go.Scatter(x=x, y=tr*multiplier, mode="lines", showlegend=False, opacity = 0.4, line={"color": colors[i%col_len]}))
    fig.update_yaxes(type="log")
    fig.show()
    
    fig = prep_fig(f"{quantity}: for lowest energy {data_x[0]} MeV")
    fig.add_trace(go.Scatter(x=x, y=pred[0]*multiplier, mode="lines", showlegend=False, opacity = 1, line={"color": colors[i%col_len]}))
    fig.add_trace(go.Scatter(x=x, y=true[0]*multiplier, mode="lines", showlegend=False, opacity = 0.4, line={"color": colors[i%col_len]}))
    fig.update_yaxes(type="log")
    fig.show()
    
    fig = prep_fig(f"{quantity}: for highest energy {data_x[-1]} MeV")
    fig.add_trace(go.Scatter(x=x, y=pred[-1]*multiplier, mode="lines", showlegend=False, opacity = 1, line={"color": colors[i%col_len]}))
    fig.add_trace(go.Scatter(x=x, y=true[-1]*multiplier, mode="lines", showlegend=False, opacity = 0.4, line={"color": colors[i%col_len]}))
    fig.update_yaxes(type="log")
    fig.show()    


plot_true_pred_log(normalized_data_z_dose_test_halves[::], test_predictions_halves['dose_z'], max_z_dose, "Dose z")



# %% [markdown]
# ### Fluence z


# %%
plot_true_pred_log(normalized_data_z_fluence_protons_test_halves[::],  test_predictions_halves['fluence_z'], max_z_fluence_protons, "Fluence z")


# %% [markdown]
# ### dLet z


# %%
plot_true_pred_log(normalized_data_z_dlet_protons_test_halves[::],  test_predictions_halves['dlet_z'], max_z_dlet_protons, "dLet z")


# %% [markdown]
# ## difference


# %% [markdown]
# ### dose z difference 


# %%
def prep_fig(title):
    fig = go.Figure()
    fig.update_layout(
        width=1200,   # px, default is 700
        height=1000,   # px, default is 450
        autosize=False,  # must be False to fix the size
        title = title,
    )
    return fig
x = list(range(400))


colors = ["red", "blue", "green", "orange"]
col_len = len(colors)


every_nth = 10
def plot_diff(true, pred, multiplier, quantity):


    metric = (pred-true)*multiplier


    fig = prep_fig(f"{quantity}: Difference for all energies")
    for i,serie in enumerate(metric):
        fig.add_trace(go.Scatter(x=x, y=serie, mode="lines", showlegend=False, opacity = 1, line={"color": colors[i%col_len]}))


    fig.show()
    
    fig = prep_fig(f"{quantity}: Difference for lowest energy")
    fig.add_trace(go.Scatter(x=x, y=metric[0], mode="lines", showlegend=False, opacity = 1, line={"color": colors[i%col_len]}))



    fig.show()
    fig = prep_fig(f"{quantity}: Difference for highest energy")
    fig.add_trace(go.Scatter(x=x, y=metric[-1], mode="lines", showlegend=False, opacity = 1, line={"color": colors[i%col_len]}))



    fig.show()    


plot_diff(normalized_data_z_dose_test_halves[::], test_predictions_halves['dose_z'], max_z_dose, "Dose z")


# %% [markdown]
# ### fluence z difference 


# %%
plot_diff(normalized_data_z_fluence_protons_test_halves[::], test_predictions_halves['fluence_z'], max_z_dlet_protons, "Fluence z")


# %% [markdown]
# ### dlet z difference 


# %%
plot_diff(normalized_data_z_dlet_protons_test_halves[::], test_predictions_halves['dlet_z'], max_z_dlet_protons, "dlet z")


# %% [markdown]
# ## Percent diff


# %% [markdown]
# ### Dose z percent difference


# %%
def print_percent_diff(true, pred, quantity):


    metric = (true/pred-1)*100


    fig = prep_fig(f"{quantity}: Percent difference for all energies")
    for i,serie in enumerate(metric[::]):
        fig.add_trace(go.Scatter(x=x, y=serie, mode="lines", showlegend=False, opacity = 1, line={"color": colors[i%col_len]}))


    fig.show()
    
    fig = prep_fig(f"{quantity}: Percent difference for lowest energy")
    fig.add_trace(go.Scatter(x=x, y=metric[0], mode="lines", showlegend=False, opacity = 1, line={"color": colors[i%col_len]}))



    fig.show()
    fig = prep_fig(f"{quantity}: Percent difference for highest energy")
    fig.add_trace(go.Scatter(x=x, y=metric[-1], mode="lines", showlegend=False, opacity = 1, line={"color": colors[i%col_len]}))



    fig.show()    


print_percent_diff(normalized_data_z_dose_test_halves[::], test_predictions_halves['dose_z'], "Dose z")


# %% [markdown]
# ### Fluence z percent difference


# %%
print_percent_diff(normalized_data_z_fluence_protons_test_halves[::], test_predictions_halves['fluence_z'], "Fluence z")


# %% [markdown]
# ### Dlet z percent difference


# %%
print_percent_diff(normalized_data_z_dlet_protons_test_halves[::], test_predictions_halves['dlet_z'], "dLet z")