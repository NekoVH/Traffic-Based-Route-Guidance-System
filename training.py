import datetime
import torch
import numpy as np
import matplotlib.pyplot as plt
from utils.data_processing import *
class Optimization:
    def __init__(self, model, loss_fn, optimizer):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.train_losses = []

    def train_step(self, x, y):
        self.model.train()

        yhat = self.model(x)

        loss = self.loss_fn(y, yhat)

        loss.backward()

        self.optimizer.step()
        self.optimizer.zero_grad()

        return rescaler(loss.item())

    def train(self, train_loader, batch_size=64, n_epochs=50, n_features=1, device='cpu'):
        for epoch in range(1, n_epochs + 1):
            batch_losses = []
            for x_batch, y_batch in train_loader:
                x_batch = x_batch.view([batch_size, -1, n_features]).to(device)
                y_batch = y_batch.to(device)
                loss = self.train_step(x_batch, y_batch)
                batch_losses.append(loss)
            training_loss = np.mean(batch_losses)
            self.train_losses.append(training_loss)

            if (epoch <= 10) | (epoch % 50 == 0):
                print(
                    f"[{epoch}/{n_epochs}] Training loss: {training_loss:.4f}\t"
                )


    def evaluate(self, test_loader, batch_size=1, n_features=1, device="cpu"):
        with torch.no_grad():
            predictions = []
            values = []
            for x_test, y_test in test_loader:
                x_test = x_test.view([batch_size, -1, n_features]).to(device)
                y_test = y_test.to(device)
                self.model.eval()
                yhat = self.model(x_test)
                predictions.append(yhat.to(device).detach().numpy())
                values.append(y_test.to(device).detach().numpy())

        return predictions, values

    def plot_losses(self):
        plt.plot(self.train_losses, label="Training loss")
        plt.legend()
        plt.title("Losses")
        plt.show()
        plt.close()
