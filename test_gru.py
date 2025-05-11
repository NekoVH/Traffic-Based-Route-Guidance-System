import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from utils.data_processing import read_excel, process_data
from gru import GRU
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
X_test = torch.tensor(flow_dataset.X_test, dtype=torch.float32)
y_test = torch.tensor(flow_dataset.y_test, dtype=torch.float32)

train_ds = TensorDataset(X_train, y_train)
test_ds = TensorDataset(X_test, y_test)

train_dl = DataLoader(train_ds, batch_size=64, drop_last=True)
test_dl = DataLoader(test_ds, batch_size=64, drop_last=True)

# Model parameters
input_dim = 7
output_dim = 1
hidden_dim = 64
layer_dim = 1
batch_size = 64
dropout = 0.2
n_epochs = 1000
learning_rate = 1e-3
weight_decay = 1e-6

# Init model, loss and optimiser
model = GRU().to(device)
loss_func = nn.MSELoss()
optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

opt = Optimization(model=model, loss_fn=loss_func, optimizer=optimiser, rescaler=flow_rescaler, device=device)
opt.train(train_dl, n_epochs=n_epochs, n_features=input_dim)
opt.plot_losses()