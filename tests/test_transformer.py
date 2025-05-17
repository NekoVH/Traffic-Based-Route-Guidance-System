import torch
from torch import optim, nn
from torch.utils.data import TensorDataset, DataLoader

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from training import Optimization
from transformer import Transformer
from utils.data_processing import process_data, read_excel

from hyperparams import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
flow_dataset, flow_rescaler = process_data(read_excel("datasets/Scats Data October 2006.xls", sheet_name="Data", header=1))

X_train = torch.tensor(flow_dataset.X_train, dtype=torch.float32)
X_val = torch.tensor(flow_dataset.X_val, dtype=torch.float32)
X_test = torch.tensor(flow_dataset.X_test, dtype=torch.float32)

y_train = torch.tensor(flow_dataset.y_train, dtype=torch.float32)
y_val = torch.tensor(flow_dataset.y_val, dtype=torch.float32)
y_test = torch.tensor(flow_dataset.y_test, dtype=torch.float32)

training_data = TensorDataset(X_train, y_train)
validation_data = TensorDataset(X_val, y_val)
test_data = TensorDataset(X_test, y_test)

# Load Dataset to DataLoader
train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=False, drop_last=True)
validation_dataloader = DataLoader(validation_data, batch_size=batch_size, shuffle=False, drop_last=True)
test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False, drop_last=True)

# Initialize model with memory optimizations
model = Transformer().to(device)

loss_fn = torch.nn.MSELoss(reduction='mean')
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

opt = Optimization(
    model=model,
    optimizer=optimizer,
    loss_fn=loss_fn,
    epochs=n_epochs,
    rescaler=flow_rescaler,
    device=device,
)

opt.train(train_dataloader, validation_dataloader, n_features=input_dim)
opt.plot_losses()

opt.evaluate(test_dataloader, n_features=input_dim)

