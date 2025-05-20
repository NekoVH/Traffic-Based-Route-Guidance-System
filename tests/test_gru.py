import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from hyperparams import *
from gru import GRU
from utils.data_processing import read_excel, process_data
from training import Optimization

# Reference: https://pytorch.org/tutorials/beginner/basics/optimization_tutorial.html

# Determine device and init dataset
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
flow_dataset, flow_rescaler = process_data(read_excel(
    filename="datasets/Scats Data October 2006.xls",
    sheet_name="Data",
    header=1
))

X_train = torch.tensor(flow_dataset.X_train, dtype=torch.float32)
y_train = torch.tensor(flow_dataset.y_train, dtype=torch.float32)

X_val = torch.tensor(flow_dataset.X_val, dtype=torch.float32)
y_val = torch.tensor(flow_dataset.y_val, dtype=torch.float32)

X_test = torch.tensor(flow_dataset.X_test, dtype=torch.float32)
y_test = torch.tensor(flow_dataset.y_test, dtype=torch.float32)

train_ds = TensorDataset(X_train, y_train)
val_ds = TensorDataset(X_val, y_val)
test_ds = TensorDataset(X_test, y_test)

train_dl = DataLoader(train_ds, batch_size=batch_size, drop_last=True)
val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=True)
test_dl = DataLoader(test_ds, batch_size=batch_size, drop_last=True)

# Init model, loss and optimiser
model = GRU().to(device)
loss_func = nn.MSELoss()
optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

opt = Optimization(model=model, loss_fn=loss_func, epochs=n_epochs, optimizer=optimiser, rescaler=flow_rescaler, device=device)
opt.train(train_dl, val_loader=val_dl, n_features=input_dim)
opt.plot_losses()
