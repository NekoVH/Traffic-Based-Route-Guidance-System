# Traffic-Based Route Guidance System (TBRGS)

## Overview

This project is part of COS30019 - Introduction to AI, Assignment 2 Part B. It focuses on implementing a Machine Learning-powered Traffic-Based Route Guidance System (TBRGS) that predicts traffic flow and integrates the prediction into an optimal route guidance tool for the City of Boroondara in Melbourne, Australia.

The system uses deep learning models (LSTM and GRU) and one additional model of our choice (currently 2016 traditional transformer) to predict traffic conditions. The results are used to estimate travel time between intersections and recommend up to five optimal routes between a user-defined origin and destination.

## Features
- Traffic Flow Prediction using:
	- Long Short-Term Memory (LSTM)
	- Gated Recurrent Unit (GRU)
	- Transformer
- Route Optimisation based on predicted travel times
- Evaluation and COmparison of different ML models
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
6. GUI: Provides an interactive (?) interface for users.
