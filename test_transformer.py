import torch
from torch import optim, nn
from torch.utils.data import TensorDataset, DataLoader

from training import Optimization
from transformer import Transformer
from utils.data_processing import process_data, read_excel

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

# Hyperparameters
input_dim = 7
output_dim = 1
hidden_dim = 64
layer_dim = 1
batch_size = 64
n_epochs = 1000
learning_rate = 1e-3
weight_decay = 1e-6

# Load Dataset to DataLoader
# Transformer expects (batch, seq_len, input_dim)
train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=False, drop_last=True)
validation_dataloader = DataLoader(validation_data, batch_size=batch_size, shuffle=False, drop_last=True)
test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False, drop_last=True)

model = Transformer().to(device)

loss_fn = nn.MSELoss(reduction="mean")
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

opt = Optimization(model=model, loss_fn=loss_fn, optimizer=optimizer, rescaler=flow_rescaler)
opt.train(train_dataloader, validation_dataloader, n_epochs=n_epochs, n_features=input_dim)
opt.plot_losses()

