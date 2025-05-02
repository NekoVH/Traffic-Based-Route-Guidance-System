from models import *

model = LSTM()

inputs = torch.tensor([[0.0, 0.5, 0.25, 1.0], [1.0, 0.5, 0.25, 1.0]])
labels = torch.tensor([0.0, 1.0])

dataset = TensorDataset(inputs, labels)
dataloader = DataLoader(dataset)

trainer = L.Trainer(max_epochs=3000)
trainer.fit(model, train_dataloaders=dataloader)


print("\nNow let's compare the observed and predicted values...")
print("Company A: Observed = 0, Predicted =", model(torch.tensor([0., 0.5, 0.25, 1.])).detach())
print("Company B: Observed = 1, Predicted =", model(torch.tensor([1., 0.5, 0.25, 1.])).detach())
