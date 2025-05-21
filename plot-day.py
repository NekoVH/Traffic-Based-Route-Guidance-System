import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils.data_processing import process_data, get_avg_data
from gru import GRU
from lstm import LSTM
from transformer import Transformer

from sys import argv
import os

def parse_args():
    argc = len(argv)
    if argc != 2:
        print("Usage: python3 day_prediction.py <model choice>")
        print("Models currently supported:")
        print("\tlstm - Long Short-Term Memory")
        print("\tgru - Gated Recurrent Unit")
        print("\ttransformer")
        exit(1)
    else:
        model = argv[1].lower()
        return model


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


def get_site_data(lat, long):
    all_data = get_avg_data()

    # This uses exact match right now, but we might need to use something like np.isclose()
    return all_data[(all_data['NB_LATITUDE'] == lat) & (all_data['NB_LONGITUDE'] == long)]


def predict_24_hours(model, site_data, scaler, device="cpu"):
    time_cols = site_data.columns[2:]   # Skip lat/long columns
    raw_values = site_data[time_cols].values.flatten().astype(float)

    scaled_vals = scaler(raw_values)
    predictions = []

    model.eval()
    with torch.no_grad():
        for i in range(len(scaled_vals) - 7):
            inputs = scaled_vals[i:i + 7]   # Get previous 7 lags
            input_tensor = torch.tensor(
                inputs,
                dtype=torch.float32,
                device=device
            ).unsqueeze(0).unsqueeze(1)     # [batch of 1, 1 feature, 7 time steps]
                
            prediction = model(input_tensor).squeeze().item()
            predictions.append(prediction)
    
    return np.array(predictions), raw_values


def calculate_speeds(flow_rates):
    # Assumptions: speed limit = 60km/h, road capacity = 1500 vehicles/hr
    # flow = -1.4648375*(speed)^2 + 93.75*(speed)
    # 1.4648375*(speed)^2 - 93.75*(speed) + flow = 0     # Use quadratic formula to get speed

    # Under capacity -> <1500 vehicles                  # Consider upper speeds
    # Over capacity -> >1500 vehicles                   # Consider lower speeds
    
    a = 1.4648375
    b = -93.75

    speeds = []
    for flow in flow_rates:
        c = flow
        is_overcapacity = False

        if c > 1500:
            is_overcapacity = True
            c = 3000 - flow

        discriminant = b**2 - 4*a*c

        if discriminant < 0:
            speeds.append(0)    # Do not handle complex solutions
            continue

        sqrt_discriminant = np.sqrt(discriminant)
        speed1 = (-b + sqrt_discriminant) / (2 * a)
        speed2 = (-b - sqrt_discriminant) / (2 * a)

        if is_overcapacity:
            speed = max(min(speed1, speed2), 0)         # Take lower speed
        else:
            if c <= 351:
                speed = min(max(speed1, speed2), 60)    # Take higher speed, 60km/h cap
            else:
                speed = max(max(speed1, speed2), 0)     # Take higher speed

        speeds.append(speed)
        print(f"({flow}, {speed})")

    return speeds


def plot_flows(
    predictions,
    actual,
    time_labels,
    title="24h Traffic Flow Prediction",
    num_points=96
):
    plt.figure(figsize=(14, 7))
    plt.plot(actual[:num_points], label="Actual Flow", linewidth=2)
    plt.plot(range(7, 7 + len(predictions[:num_points - 7])), predictions[:num_points - 7], label="Predicted Flow", linestyle="--")
    
    plt.xticks(ticks=range(num_points), labels=time_labels, rotation=60)
    plt.xlabel("Time")
    plt.ylabel("Traffic flow (vehicles / hour)")
    plt.title(title)
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()


def plot_speeds(
    predictions,
    actual,
    time_labels,
    title="24h Traffic Speed Prediction",
    num_points=96
):
    plt.figure(figsize=(14, 7))
    plt.plot(actual[:num_points], label="Actual Speed", linewidth=2)
    plt.plot(range(7, 7 + len(predictions[:num_points - 7])), predictions[:num_points - 7], label="Predicted Speed", linestyle="--")
    
    plt.xticks(ticks=range(num_points), labels=time_labels, rotation=60)
    plt.xlabel("Time")
    plt.ylabel("Speed (km/h)")
    plt.title(title)
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Init model
    model_choice = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(model_choice, device)

    # Obtain actual flow from site and 
    lat, long = (-37.86703, 145.09159)   # SCATs site 0970: WARRIGAL_RD N of HIGH STREET_RD
    site_data = get_site_data(lat, long)
    _, rescaler, scaler = process_data()

    flow_predictions_scaled, flow_actual = predict_24_hours(model, site_data, scaler, device=device)
    flow_predictions = rescaler(flow_predictions_scaled)

    # Scale from vehicles/15mins -> vehicles/hour
    flow_predictions *= 4
    flow_actual *= 4

    speed_predictions = calculate_speeds(flow_predictions)
    speed_actual = calculate_speeds(flow_actual)

    # Create time labels from 00:00 - 23:45
    time_labels = pd.date_range(start="00:00", periods=96, freq="15min").strftime('%H:%M')

    # Plot traffic flow predictions
    plot_flows(
        flow_predictions,
        flow_actual,
        time_labels,
        title=f"0970: WARRIGAL_RD N of HIGH STREET_RD ({lat}, {long}) Flow Prediction - {model_choice.upper()}"
    )

    # Plot speed predictions
    plot_speeds(
        speed_predictions,
        speed_actual,
        time_labels,
        title=f"0970: WARRIGAL_RD N of HIGH STREET_RD ({lat}, {long}) Speed Prediction - {model_choice.upper()}"
    )