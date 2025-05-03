from torch.utils.data import TensorDataset, DataLoader
import torch
import lightning as L
from models import LSTM

from utils.data_processing import read_excel, process_data

model = LSTM()

df = read_excel(
    filename="datasets/Scats Data October 2006.xls",
    sheet_name="Data",
    header=1
)

flow_dataset = process_data(df)

print(f"{flow_dataset.X_train} {flow_dataset.X_train.shape}\n")
print(f"{flow_dataset.y_train} {flow_dataset.y_train.shape}")

dataset = TensorDataset(torch.tensor(flow_dataset.X_train), torch.tensor(flow_dataset.y_train))
dataloader = DataLoader(dataset, num_workers=15)

trainer = L.Trainer(max_epochs=3000)
trainer.fit(model, train_dataloaders=dataloader)
