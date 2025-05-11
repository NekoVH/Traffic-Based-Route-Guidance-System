from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import torch
from models import LSTM

from utils.data_processing import read_excel, process_data

lstm = LSTM()

df = read_excel(
    filename="datasets/Scats Data October 2006.xls",
    sheet_name="Data",
    header=1
)

flow_dataset = process_data(df)

X_train = torch.FloatTensor(flow_dataset.X_train)
X_test = torch.FloatTensor(flow_dataset.X_test)
y_train = torch.FloatTensor(flow_dataset.y_train)
y_test = torch.FloatTensor(flow_dataset.y_test)

print(f"{X_train} {X_train.shape}\n")
print(f"{y_train} {y_train.shape}")
