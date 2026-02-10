# Jakarta Temperature Forecasting with LSTM

This project implements a **Long Short-Term Memory (LSTM)** deep learning model to forecast **daily temperature in Jakarta** using historical weather data. The notebook covers the full workflow from data preprocessing and exploration to model training and hyperparameter optimization.

---

## Overview

Time series forecasting is commonly used in weather prediction to capture temporal patterns and seasonal trends.  
This project uses historical meteorological data to train an LSTM model capable of learning long-term dependencies and predicting future temperature values.

The workflow includes:
- Data cleaning and preparation  
- Exploratory Data Analysis (EDA)  
- Time series sequence generation  
- Model training and evaluation  
- Hyperparameter tuning using **Bayesian Optimization**

---

## Features

- Time series forecasting using **TensorFlow/Keras LSTM**
- Data preprocessing:
  - Date parsing and indexing
  - Filtering inconsistent or unnecessary data
  - Feature and target preparation
- **MinMaxScaler** normalization for stable training
- Sequence window generation for LSTM input
- **Train–test split (80/20)**
- Visualization of temperature trends
- Model performance evaluation using:
  - Mean Squared Error (MSE)
  - Root Mean Squared Error (RMSE)
- **Hyperparameter tuning** with:
  - Keras Tuner
  - Bayesian Optimization
- Early stopping and model checkpointing to prevent overfitting

---

## Dataset

- Historical weather data for Jakarta
- Time range: approximately **2019–2024** (after filtering)
- Target variable:  
  **`temperature_2m`** (daily temperature)

---

## Model Architecture

- LSTM layers for temporal pattern learning
- Dense output layer for regression
- Tuned hyperparameters include:
  - Number of LSTM units
  - Optimizer selection (Adam, RMSprop, Nadam)
  - Learning rate
- Best configuration selected based on **validation loss**

---

## Tech Stack

- Python
- TensorFlow / Keras
- Keras Tuner
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

---

## Project Structure

├── main.ipynb # Main notebook (full pipeline)
├── jakarta5years.csv # Weather dataset
├── requirements.txt # Project dependencies
└── README.md

## Results

The trained LSTM model learns temporal weather patterns and produces temperature predictions evaluated using RMSE and MSE of ~0,8 °C of error.
Hyperparameter tuning with Bayesian Optimization improves performance while reducing manual experimentation.
