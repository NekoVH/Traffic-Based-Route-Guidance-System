import torch
import numpy as np
from math import atan2, radians, sin, cos, sqrt # For calculating edge length/distance 

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from lstm import LSTM
from gru import GRU
from transformer import Transformer
from utils.data_processing import process_data, get_avg_data


# Function to calculate length of road segments
def haversine(lat1, lon1, lat2, lon2):
    # Radius of the Earth in kilometers
    R = 6371
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])

    # Haversine formula to calculate distance
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    distance = R * c

    return round(distance, 4) # The result is in kilometers

def reconstruct_path(node: str, prev: list):
    """Given a node, backtrack through the prev list to see the path taken"""
    path = []
    iter = node
    while iter is not None:
        path.append(iter)
        iter = prev[iter]
    return list(reversed(path))


# Speed prediction class
class SpeedPredictor:
    """Class to return predicted speeds"""
    def __init__(self, model_choice="lstm", device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _, self.rescaler, self.scaler = process_data()
        self.avg_flows = get_avg_data()

        self.model = None
        self.set_model(model_choice.lower())

    def set_model(self, model_choice: str):
        """Load and switch the model to the specified type"""
        choice = model_choice.lower()
        
        SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
        MODEL_PATH = os.path.join(SCRIPT_DIR, f"../models/{choice}.pth")

        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model weights not found at {MODEL_PATH}")
        
        if choice == "lstm":
            model = LSTM()
        elif choice == "gru":
            model = GRU()
        elif choice == "transformer":
            model = Transformer()
        else:
            raise ValueError(f"Unsupported model: {model_choice}")

        model.load_state_dict(torch.load(MODEL_PATH, map_location=self.device))
        self.model = model.to(self.device).eval()

    def _calculate_speed(self, flow):
        """
        Convert flow (vehicles/hr) into estimated speed (km/h).\n
        Assumptions: speed limit = 60km/h, road capacity = 1500 vehicles/hr\n
        flow = -1.4648375*(speed)^2 + 93.75*(speed)\n
        1.4648375*(speed)^2 - 93.75*(speed) + flow = 0 -> Use quadratic formula to get speed\n
        Under capacity -> <1500 vehicles\n
        Over capacity -> >1500 vehicles\n
        """
        
        a = 1.4648375
        b = -93.75
        c = flow

        is_overcapacity = c > 1500
        if is_overcapacity: c = 3000 - c # Mirror so we do not get undefined solutions

        discriminant = b**2 - 4*a*c

        if discriminant < 0:
            return 0    # Do not handle complex solutions

        sqrt_discriminant = np.sqrt(discriminant)
        speed1 = (-b + sqrt_discriminant) / (2 * a)
        speed2 = (-b - sqrt_discriminant) / (2 * a)

        if is_overcapacity:
            return max(min(speed1, speed2), 0)         # Take lower speed
        else:
            if flow <= 351:
                return min(max(speed1, speed2), 60)    # Take higher speed, 60km/h cap
            else:
                return max(max(speed1, speed2), 0)     # Take higher speed, 0km/h if negative

    def get_speed(self, lat, long, time: str):
        """Predict the traffic speed given a lat, long, and time."""
        
        # Obtain avg flows from site
        site_data = self.avg_flows[
            (self.avg_flows['NB_LATITUDE'] == lat) &
            (self.avg_flows['NB_LONGITUDE'] == long)
        ]
        if site_data.empty:
            # Above uses exact match, but we might need to use something like np.isclose()
            raise ValueError(f"No data found for location ({lat}, {long})")
        
        time_cols = site_data.columns[2:]   # Skip lat, long columns
        if time not in time_cols:
            raise ValueError(f"Time '{time}' is not a valid time choice.")

        time_index = list(time_cols).index(time)

        # Get 7 flow values before the selected time as the lag window
        raw_flows = site_data[time_cols].values.flatten().astype(float)
        scaled_flows = self.scaler(raw_flows)
        scaled_flows_ext = np.concatenate([scaled_flows[-7:], scaled_flows])    # Prepend last 7 values for lag window at start
        input = scaled_flows_ext[time_index:time_index + 7]

        input_tensor = torch.tensor(input, dtype=torch.float32, device=self.device)
        input_tensor = input_tensor.unsqueeze(0).unsqueeze(1)   # Ensure the shape is correct

        # Get flow prediction from model
        with torch.no_grad():
            flow_pred_scaled = self.model(input_tensor).squeeze().item()

        flow_pred_rescaled = self.rescaler(flow_pred_scaled) * 4

        return self._calculate_speed(flow_pred_rescaled)