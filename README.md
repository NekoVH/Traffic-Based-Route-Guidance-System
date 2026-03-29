# Traffic-Based Route Guidance System (TBRGS)

## Overview

This project implements a Machine Learning-powered Traffic-Based Route Guidance System (TBRGS) that predicts traffic flow and integrates the prediction into an optimal route guidance tool for the City of Boroondara in Melbourne, Australia.

The system uses deep learning models (LSTM and GRU) and one additional model of our choice (currently 2016 traditional transformer) to predict traffic conditions. The results are used to estimate travel time between intersections and recommend up to five optimal routes between a user-defined origin and destination.

## Features
- Traffic Flow Prediction using:
	- Long Short-Term Memory (LSTM)
	- Gated Recurrent Unit (GRU)
	- Transformer
- Route Optimisation based on predicted travel times
- Evaluation and Comparison of different ML models
- Graphical User Interface (GUI) for:
	- Inputting origin/destinations (SCATS site numbers)
	- Viewing predictions and travel time estimates
	- Adjusting parameters and running predictions
- Configuration File support for default settings

## Dataset
The system uses traffic flow data provided by VicRoads for the City of Boroondara (October 2006). The data includes car counts at intersections every 15 minutes.

## How it works
1. Data Preprocessing: Extracts and formats the dataset into appropriate structures for ML training
2. Model Training: Trains three different ML models on the dataset
3. Prediction: Estimates traffic flow for future time intervals
4. Travel Time Estimation: Converts flow predictions into travel time using simplified assumptions.
5. Pathfinding: Integrates with Assignment 2A to find the top-k optimal routes (k <= 5) between two SCATS sites.
6. GUI: Provides an interactive interface for users.

## Using the GUI

### Starting the Application
To start the GUI application, run:
```bash
python gui/gui.py
```

### Features and Usage

#### Basic Navigation
- The GUI displays a map of SCATS intersections in Boroondara
- Enter source and destination SCATS numbers in the input fields
- Click "Calculate Paths" to find optimal routes

#### Path Finding Modes
1. **Shortest Path Mode**
   - Finds routes with the shortest total distance
   - Shows distances between intersections
   - Toggle "Show Distances" to display edge distances on the map

2. **Fastest Path Mode**
   - Finds routes with the shortest predicted travel time
   - Requires selecting:
     - Time of day (15-minute intervals)
     - AI model (LSTM, GRU, or Transformer)
   - Shows predicted travel times between intersections
   - Toggle "Show Times" to display edge travel times on the map

#### Results Display
- Up to 5 optimal routes are shown in the results panel
- Each route shows:
  - Complete path through SCATS intersections
  - Total distance (in km) or time (in minutes)
- Click on any route to highlight it on the map
- Hover over routes for better visibility

### Tips
- Use the scrollbars in the results panel to view all routes
- Switch between distance and time display using the checkboxes
- The fastest path mode provides more realistic travel times by considering traffic conditions
- Different AI models may give slightly different predictions for the same route
