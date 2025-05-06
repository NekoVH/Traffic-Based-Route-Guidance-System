import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from gru import GRU

# Reference: https://pytorch.org/tutorials/beginner/basics/optimization_tutorial.html

# Synthetic dataset with sets of random nums, target = sum of random nums in set
class SumDataset(Dataset):
    def __init__(self, num_samples=1000, seq_len=10):
        self.x = torch.randn(num_samples, seq_len, 1)   # Dataset
        self.y = self.x.sum(dim=1)                      # Target

    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, index):
        return self.x[index], self.y[index]


# Determine device and init dataset
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataset = SumDataset(num_samples=3000, seq_len=10)

print("Random Numbers Dataset:")
for i, data in enumerate(dataset):
    x_sample, y_sample = data
    num_set = [round(val, 4) for val in x_sample.squeeze().tolist()]
    target = round(y_sample.item(), 4)
    print(f"Sample {i + 1} - x: {num_set}, y: {target}")
print("--------------------------------------------------------------------------------------------------------------------")

# Partition dataset into smaller non-overlapping datasets
# 70% train, 15% validation, 15% test
ds_length = len(dataset)
train_length = int(ds_length * 0.7)
val_length = int(ds_length * 0.15)
test_length = ds_length - train_length - val_length
train_ds, val_ds, test_ds = random_split(dataset, [train_length, val_length, test_length])

train_dl = DataLoader(train_ds, batch_size=32, shuffle=True)
val_dl = DataLoader(val_ds, batch_size=32)
test_dl = DataLoader(test_ds, batch_size=32)

# Init model, loss and optimiser
model = GRU().to(device)
criterion = nn.MSELoss()
optimiser = torch.optim.Adam(model.parameters(), lr=1e-3)

# Training loop with validation checking
for epoch in range(1, 21):
    # Train
    model.train()   # Enable dropout (0.2 is set for GRU)
    train_loss = 0.0
    
    for xb, yb in train_dl:
        xb, yb = xb.to(device), yb.to(device)   # Move batch to device
        
        # Calculate prediction and loss
        preds = model(xb)                       # Forward pass
        loss = criterion(preds, yb)             # Compute MSE

        # Backpropagation
        loss.backward()                         # Backpropagate
        optimiser.step()                        # Update weights
        optimiser.zero_grad()                   # Clear old gradients
        
        # Add loss to sum
        train_loss += loss.item()
    
    train_loss /= len(train_dl)                 # Calculate average loss per batch

    # Validate
    model.eval()    # Disable dropout
    val_loss = 0.0
    with torch.no_grad():   # Ensures no gradients computed
        for xb, yb in val_dl:
            xb, yb = xb.to(device), yb.to(device)
            val_loss += criterion(model(xb), yb).item()
    val_loss /= len(val_dl)

    print(f"Epoch {epoch:02d} — train MSE: {train_loss:.4f}, val MSE: {val_loss:.4f}")

# Final evaluation using the test dataset
model.eval()
test_loss = 0.0
with torch.no_grad():
    for xb, yb in test_dl:
        xb, yb = xb.to(device), yb.to(device)
        test_loss += criterion(model(xb), yb).item()
test_loss /= len(test_dl)

print(f"\nTest set MSE: {test_loss:.4f}")