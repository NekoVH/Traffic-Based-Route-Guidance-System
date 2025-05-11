import datetime
import copy
import torch
import numpy as np
import matplotlib.pyplot as plt
from utils.data_processing import *

class Optimization:
    def __init__(self, model, loss_fn, optimizer, rescaler, device='cpu',
                 early_exit_patience=20, lr_scheduler_patience=10, lr_factor=0.5):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", patience=lr_scheduler_patience, factor=lr_factor
        )
        self.rescaler = rescaler
        self.train_losses = []
        self.val_losses = []
        self.early_exit_patience = early_exit_patience
        self.device = device


    def train_step(self, x, y):
        self.model.train()

        yhat = self.model(x)

        # Reshape yhat if required
        if yhat.shape != y.unsqueeze(1).shape:
            yhat = yhat.view_as(y.unsqueeze(1))

        loss = self.loss_fn(y.unsqueeze(1), yhat)

        loss.backward()

        self.optimizer.step()
        self.optimizer.zero_grad()

        return self.rescaler(loss.item())

    def validation_step(self, x, y):
        self.model.eval()
        with torch.no_grad():
            yhat = self.model(x)
            if yhat.shape != y.unsqueeze(1).shape:
                yhat = yhat.view_as(y.unsqueeze(1))
            loss = self.loss_fn(y.unsqueeze(1), yhat)
        return self.rescaler(loss.item())

    def train(self, train_loader, val_loader, batch_size=64, n_epochs=50, n_features=1):
        best_val_loss = float('inf')
        best_weights = copy.deepcopy(self.model.state_dict())
        no_improvement_count = 0
        
        for epoch in range(1, n_epochs + 1):
            # Training phase
            self.model.train()
            train_batch_losses = []
            for x_batch, y_batch in train_loader:
                x_batch = x_batch.view([batch_size, -1, n_features]).to(self.device)
                y_batch = y_batch.to(self.device)
                loss = self.train_step(x_batch, y_batch)
                train_batch_losses.append(loss)
            training_loss = np.mean(train_batch_losses)
            self.train_losses.append(training_loss)

            # Validation phase
            self.model.eval()
            val_batch_losses = []
            for x_val, y_val in val_loader:
                x_val = x_val.view([batch_size, -1, n_features]).to(self.device)
                y_val = y_val.to(self.device)
                loss = self.validation_step(x_val, y_val)
                val_batch_losses.append(loss)
            validation_loss = np.mean(val_batch_losses)
            self.val_losses.append(validation_loss)

            # Learning rate scheduling based on validation loss
            self.scheduler.step(validation_loss)

            # Early stopping based on validation loss
            if validation_loss < best_val_loss:
                best_val_loss = validation_loss
                best_weights = copy.deepcopy(self.model.state_dict())
                no_improvement_count = 0
            else:
                no_improvement_count += 1

            if (epoch <= 10) | (epoch % 50 == 0):
                print(
                    f"[{epoch}/{n_epochs}] Training loss: {training_loss:.4f}"
                    f" - Validation loss: {validation_loss:.4f}"
                )
            
            if no_improvement_count >= self.early_exit_patience:
                print(f"Early exit triggered at epoch {epoch} with best validation loss: {best_val_loss:.4f}")
                break
        
        # Load best model weights
        self.model.load_state_dict(best_weights)
        return best_val_loss

    def evaluate(self, test_loader, batch_size=1, n_features=1):
        self.model.eval()
        predictions = []
        values = []
        test_losses = []
        
        with torch.no_grad():
            for x_test, y_test in test_loader:
                x_test = x_test.view([batch_size, -1, n_features]).to(self.device)
                y_test = y_test.to(self.device)
                
                yhat = self.model(x_test)
                if yhat.shape != y_test.unsqueeze(1).shape:
                    yhat = yhat.view_as(y_test.unsqueeze(1))
                
                # Calculate test loss
                loss = self.loss_fn(y_test.unsqueeze(1), yhat)
                test_losses.append(self.rescaler(loss.item()))
                
                predictions.append(yhat.cpu().numpy())
                values.append(y_test.cpu().numpy())

        test_loss = np.mean(test_losses)
        print(f"Test Loss: {test_loss:.4f}")
        
        return np.array(predictions), np.array(values), test_loss

    def plot_losses(self):
        plt.figure(figsize=(10, 6))
        plt.plot(self.train_losses, label="Training loss")
        plt.plot(self.val_losses, label="Validation loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training and Validation Losses")
        plt.legend()
        plt.grid(True)
        plt.show()
        plt.close()
