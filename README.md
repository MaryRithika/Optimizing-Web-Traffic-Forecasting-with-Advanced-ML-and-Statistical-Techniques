# Optimizing Web Traffic Forecasting with Advanced ML and Statistical Techniques 

This project demonstrates the use of advanced statistical and machine learning models to forecast web traffic. Specifically, the project implements models like SARIMA (Seasonal Autoregressive Integrated Moving Average) and compares their performance against machine learning models such as MLP (Multi-Layer Perceptron), RNN (Recurrent Neural Network), and SAE (Stacked Autoencoder). The primary goal is to predict future web traffic trends using time series data, enabling effective web traffic management.

## Features
Implementation of SARIMA and various machine learning models for time series forecasting.
Data preprocessing and analysis of web traffic datasets.
Comparison of model performances using metrics like Root Mean Squared Error (RMSE).
Visualizations showing the actual vs. predicted traffic trends.
Models Implemented
SARIMA (Seasonal Autoregressive Integrated Moving Average)
MLP (Multi-Layer Perceptron)
RNN (Recurrent Neural Network)
SAE (Stacked Autoencoder)

Dataset
The dataset used in this project consists of hourly web traffic records, with two primary columns:

Hour Index: Sequential numerical index representing the hour of the day.

Sessions: The number of user interactions (sessions) during each hour.
This dataset helps model and predict online traffic trends using time series analysis.

## Setup and Requirements
To run this project, you'll need the following dependencies:

Python 3.x
Pandas: Data manipulation library.
NumPy: Support for large, multi-dimensional arrays and matrices.
statsmodels: Statistical models and tests (for SARIMA).
matplotlib: Visualization library for plotting graphs.
scikit-learn: Machine learning library.
Install the required libraries by running:

```
pip install pandas numpy statsmodels matplotlib scikit-learn
``` 

Running the Project
Data Preparation: Ensure the web traffic dataset (in CSV format) is placed in the same directory as the script, or update the script with the correct path to your data file.
Run the Script: Execute the Python script to train the model and visualize the results.
```
python web_traffic_forecasting.py
```

## Model Performance
The project compares different models and reports performance using metrics like Root Mean Squared Error (RMSE). For instance, SARIMA achieved an accuracy of 89% on the dataset.

## Visualizations
Graphs are generated to compare actual and predicted web traffic, providing insights into the model’s forecasting performance.

### Conclusion
This project provides a comprehensive approach to web traffic forecasting by employing various statistical and machine learning techniques. Each model offers unique advantages for time series forecasting, with SARIMA excelling in accuracy and interpretability, while MLP, RNN, and SAE explore deeper learning approaches for capturing complex patterns.

### Contributors

PVS Sampath Vinayak - Data Preprocessing and Analysis
Mary Rithika Reddy Gade - Model Training and Evaluation
Akhil Nakka - Model Training and Evaluation
Manasa Motepalli - Visualization and Report Compilation
