import os
import copy
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error, r2_score, explained_variance_score
from utils.data_processing import *


class Optimization:
    def __init__(self, model, loss_fn, epochs, optimizer, rescaler, device='cpu',
                 early_exit_patience=20, lr_scheduler_patience=10, lr_factor=0.5):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.epochs = epochs
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", patience=lr_scheduler_patience, factor=lr_factor
        )
        self.rescaler = rescaler
        self.train_losses = []
        self.val_losses = []
        self.early_exit_patience = early_exit_patience
        self.device = device
        # Initialize metric history
        self.metric_history = {
            'train': {'mse': [], 'mae': [], 'smape': [], 'r2': [], 'explained_variance': []},
            'val': {'mse': [], 'mae': [], 'smape': [], 'r2': [], 'explained_variance': []}
        }

    def train_step(self, x, y):
        self.model.train()

        yhat = self.model(x)

        # Reshape yhat if required
        if yhat.shape != y.unsqueeze(1).shape:
            yhat = yhat.view_as(y.unsqueeze(1))

        # Get the scaling factor
        scale_factor = self.rescaler(1.0) - self.rescaler(0.0)
        min_value = self.rescaler(0.0)
        
        # Rescale values while preserving gradient
        y_true_scaled = y.unsqueeze(1) * scale_factor + min_value
        y_pred_scaled = yhat * scale_factor + min_value

        # Calculate loss on scaled values
        loss = self.loss_fn(y_true_scaled, y_pred_scaled)

        loss.backward()

        self.optimizer.step()
        self.optimizer.zero_grad()

        return loss.item()

    def validation_step(self, x, y):
        self.model.eval()
        with torch.no_grad():
            yhat = self.model(x)
            if yhat.shape != y.unsqueeze(1).shape:
                yhat = yhat.view_as(y.unsqueeze(1))
                
            # Get the scaling factor
            scale_factor = self.rescaler(1.0) - self.rescaler(0.0)
            min_value = self.rescaler(0.0)
            
            # Rescale values
            y_true_scaled = y.unsqueeze(1) * scale_factor + min_value
            y_pred_scaled = yhat * scale_factor + min_value
            
            # Calculate loss on scaled values
            loss = self.loss_fn(y_true_scaled, y_pred_scaled)
            
        return loss.item()

    def calculate_metrics(self, y_true, y_pred):
        """Calculate all evaluation metrics."""
        y_true_np = y_true.cpu().numpy()
        y_pred_np = y_pred.cpu().numpy()
        
        y_true_original = self.rescaler(y_true_np)
        y_pred_original = self.rescaler(y_pred_np)
        
        def smape(y_true, y_pred):
            return 2.0 * np.mean(np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred))) * 100

        metrics = {
            'mse': mean_squared_error(y_true_original, y_pred_original),
            'mae': mean_absolute_error(y_true_original, y_pred_original),
            'smape': smape(y_true_original, y_pred_original),
            'r2': r2_score(y_true_original, y_pred_original),
            'explained_variance': explained_variance_score(y_true_original, y_pred_original)
        }
        return metrics

    def train(self, train_loader, val_loader, batch_size=64, n_features=1, file=None):
        best_val_loss = float('inf')
        best_weights = copy.deepcopy(self.model.state_dict())
        no_improvement_count = 0
        
        for epoch in range(1, self.epochs+ 1):
            # Training phase
            self.model.train()
            train_batch_losses = []
            train_predictions = []
            train_targets = []
            
            for x_batch, y_batch in train_loader:
                x_batch = x_batch.view([batch_size, -1, n_features]).to(self.device)
                y_batch = y_batch.to(self.device)
                loss = self.train_step(x_batch, y_batch)
                train_batch_losses.append(loss)
                
                with torch.no_grad():
                    yhat = self.model(x_batch)
                    if yhat.shape != y_batch.unsqueeze(1).shape:
                        yhat = yhat.view_as(y_batch.unsqueeze(1))
                    train_predictions.append(yhat)
                    train_targets.append(y_batch.unsqueeze(1))
            
            training_loss = np.mean(train_batch_losses)
            self.train_losses.append(training_loss)
            
            # Calculate training metrics
            train_metrics = self.calculate_metrics(
                torch.cat(train_targets),
                torch.cat(train_predictions)
            )
            for metric_name, value in train_metrics.items():
                self.metric_history['train'][metric_name].append(value)

            # Validation phase
            self.model.eval()
            val_batch_losses = []
            val_predictions = []
            val_targets = []
            
            with torch.no_grad():
                for x_val, y_val in val_loader:
                    x_val = x_val.view([batch_size, -1, n_features]).to(self.device)
                    y_val = y_val.to(self.device)
                    loss = self.validation_step(x_val, y_val)
                    val_batch_losses.append(loss)
                    
                    yhat = self.model(x_val)
                    if yhat.shape != y_val.unsqueeze(1).shape:
                        yhat = yhat.view_as(y_val.unsqueeze(1))
                    val_predictions.append(yhat)
                    val_targets.append(y_val.unsqueeze(1))
            
            validation_loss = np.mean(val_batch_losses)
            self.val_losses.append(validation_loss)
            
            # Calculate validation metrics
            val_metrics = self.calculate_metrics(
                torch.cat(val_targets),
                torch.cat(val_predictions)
            )
            for metric_name, value in val_metrics.items():
                self.metric_history['val'][metric_name].append(value)

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
                print(f"[{epoch}/{self.epochs}]")
                print(f"Training - MSE: {train_metrics['mse']:.4f}, MAE: {train_metrics['mae']:.4f}, SMAPE: {train_metrics['smape']:.4f}, R2: {train_metrics['r2']:.4f}, Explained Variance: {train_metrics['explained_variance']:.4f}")
                print(f"Validation - MSE: {val_metrics['mse']:.4f}, MAE: {val_metrics['mae']:.4f}, SMAPE: {val_metrics['smape']:.4f}, R2: {val_metrics['r2']:.4f}, Explained Variance: {val_metrics['explained_variance']:.4f}")
            
            if no_improvement_count >= self.early_exit_patience:
                print(f"Early exit triggered at epoch {epoch} with best validation loss: {best_val_loss:.4f}")
                break
        
        # Load best model weights
        self.model.load_state_dict(best_weights)
        
        # Save model weights to file
        if (file):
            save_dir = "models"
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, file)
            torch.save(self.model.state_dict(), save_path)
            print(f"Model weights saved in {save_path}'")
        
        return best_val_loss

    def evaluate(self, test_loader, batch_size=1, n_features=1):
        self.model.eval()
        predictions = []
        values = []
        test_losses = []
        
        with torch.no_grad():
            for x_test, y_test in test_loader:
                x_test = x_test.view([x_test.shape[0], -1, n_features]).to(self.device)
                y_test = y_test.to(self.device)
                
                yhat = self.model(x_test)
                if yhat.shape != y_test.unsqueeze(1).shape:
                    yhat = yhat.view_as(y_test.unsqueeze(1))
                
                # Calculate test loss
                loss = self.loss_fn(y_test.unsqueeze(1), yhat)
                test_losses.append(self.rescaler(loss.item()))
                
                predictions.append(yhat.cpu().numpy().ravel())
                values.append(y_test.cpu().numpy().ravel())

        predictions = np.concatenate(predictions)
        values = np.concatenate(values)
        
        # Calculate all metrics
        metrics = self.calculate_metrics(
            torch.tensor(values),
            torch.tensor(predictions)
        )
        
        print("\nTest Metrics:")
        for metric_name, value in metrics.items():
            print(f"{metric_name.upper()}: {value:.4f}")
        
        return predictions, values, metrics

    def plot_losses(self):
        # Plot MSE losses
        plt.figure(figsize=(15, 10))
        
        # Plot all metrics
        metrics = ['mse', 'mae', 'smape', 'r2', 'explained_variance']
        for i, metric in enumerate(metrics, 1):
            plt.subplot(2, 3, i)
            plt.plot(self.metric_history['train'][metric], label=f"Training {metric.upper()}")
            plt.plot(self.metric_history['val'][metric], label=f"Validation {metric.upper()}")
            plt.xlabel("Epoch")
            plt.ylabel(metric.upper())
            plt.title(f"{metric.upper()}")
            plt.legend()
            plt.grid(True)
        plt.suptitle(f"Trained vs Validated Results. Epochs: {self.epochs}, Model: {self.model.__class__.__name__}")
        plt.tight_layout()
        plt.show()
        plt.close()
