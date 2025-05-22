from gui_utils import SpeedPredictor

import matplotlib.pyplot as plt
import pandas as pd

from random import randint
import os


# Select random model
choice = randint(0, 2)
if choice == 0:
    model = "lstm"
elif choice == 1:
    model = "gru"
else:
    model = "transformer"

predictor = SpeedPredictor()
predictor.set_model(model)  # Switch model

# Generate 96 time labels from 00:00 to 23:45 in 15-minute intervals
time_labels = pd.date_range("00:00", "23:45", freq="15min").strftime("%H:%M").tolist()

# Choose random location
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(SCRIPT_DIR, f"../datasets/avg_flow_data.parquet")
df = pd.read_parquet(DATA_PATH)

random_row = df.sample(n=1).squeeze()
lat = random_row['NB_LATITUDE']
long = random_row['NB_LONGITUDE']

# Get all speeds
predicted_speeds = []
for time in time_labels:
    speed = predictor.get_speed(lat, long, time)
    predicted_speeds.append(speed)

# Plot
plt.figure(figsize=(14, 7))
plt.plot(predicted_speeds, label="Predicted Speed (km/h)")
plt.xticks(ticks=range(0, 96), labels=time_labels, rotation=60)  # Show every hour
plt.title(f"{model.upper()} Predicted Traffic Speed for ({lat}, {long})")
plt.xlabel("Time of Day")
plt.ylabel("Speed (km/h)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()