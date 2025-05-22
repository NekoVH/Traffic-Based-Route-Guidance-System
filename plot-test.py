import torch
import numpy as np
import matplotlib.pyplot as plt
from utils.data_processing import read_excel, process_data
from gru import GRU
from lstm import LSTM
from transformer import Transformer

from sys import argv
import os

def parse_args():
    argc = len(argv)
    if argc != 2:
        print("Usage: python3 plot_predictions.py <model choice>")
        print("Models currently supported:")
        print("\tlstm - Long Short-Term Memory")
        print("\tgru - Gated Recurrent Unit")
        print("\ttransformer")
        exit(1)
    else:
        model = argv[1].lower()
        return model


def load_data(model, batch_size=64, device="cpu"):
    df = read_excel(
        filename="datasets/Scats Data October 2006.xls",
        sheet_name="Data", header=1
    )
    flow_dataset, flow_rescaler, _ = process_data(df)

    X_test = torch.tensor(flow_dataset.X_test, dtype=torch.float32)
    y_test = torch.tensor(flow_dataset.y_test, dtype=torch.float32)

    # We need to batch the predictions, otherwise we will get memory overflow
    predictions = []
    with torch.no_grad():
        for i in range(0, len(X_test), batch_size):
            batch = X_test[i:i + batch_size].to(device)
            if batch.shape != batch.unsqueeze(1).shape:
                batch = batch.view_as(batch.unsqueeze(1))
            prediction = model(batch).to("cpu").numpy()
            predictions.append(prediction)
    predicted = np.concatenate(predictions, axis=0)

    return predicted, y_test, flow_rescaler


def load_model(model_choice, device="cpu"):
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    WEIGHT_PATH = os.path.join(SCRIPT_DIR, f"models/{model_choice}.pth")

    if not os.path.exists(WEIGHT_PATH):
        exit(f"Error: weights file not found at {WEIGHT_PATH} first")

    if model_choice == "lstm":
        model = LSTM()
    elif model_choice == "gru":
        model = GRU()
    elif model_choice == "transformer":
        model = Transformer()
    else:
        exit(f"Error: model for {model_choice} not found")

    model.load_state_dict(torch.load(WEIGHT_PATH, map_location=device))
    return model.to(device)

def plot_predictions(model, predicted, actual, rescaler, num_points=200):
    predicted_rescaled = rescaler(predicted)
    actual_rescaled = rescaler(actual)
    
    plt.figure(figsize=(14, 7))
    plt.plot(actual_rescaled[:num_points], label="Actual Flow", linewidth=2)
    plt.plot(predicted_rescaled[:num_points], label="Predicted Flow", linestyle="--")
    plt.xlabel("15 min Increments")
    plt.ylabel("Traffic flow (vehicles)")
    plt.title(f"Traffic Flow Predictions ({model}) - First {num_points} points")
    plt.legend()
    plt.show()
    plt.close()

if __name__ == "__main__":
    # Init model
    model_choice = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(model_choice, device)

    # Get data from dataset and model
    predicted, actual, rescaler = load_data(model, batch_size=64, device=device)

    # Plot predictions
    plot_predictions(model_choice, predicted, actual, rescaler, num_points=200)
