from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import LSTM, Dense # type: ignore
import mlflow
import mlflow.keras

# Start MLflow experiment
mlflow.set_experiment("sales_prediction_experiment")

# Load the dataset
sales_data = pd.read_csv('train_data_processed.csv')

# Define features and target variable
features = sales_data.drop('Sales', axis=1)
target = sales_data['Sales']

# Data preprocessing
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(features)

# Define constants
look_back = 7  # Number of previous days to consider
forecast_horizon = 7  # Forecasting for 7 days (1 week)

# Create sequences for input and output
X, y = [], []
for i in range(len(scaled_data) - look_back - forecast_horizon):
    X.append(scaled_data[i:i + look_back])
    y.append(scaled_data[i + look_back:i + look_back + forecast_horizon, 0])  # Extract target column

X = np.array(X)
y = np.array(y)

# Split the data into training and testing sets
split_index = int(len(X) * 0.8)  # 80% training, 20% testing
X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

# Build the LSTM model
model = Sequential()
model.add(LSTM(50, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dense(forecast_horizon))  # Output layer with forecast horizon size
model.compile(optimizer='adam', loss='mse')

# Log parameters
mlflow.log_param("look_back", look_back)
mlflow.log_param("forecast_horizon", forecast_horizon)

# Train the model and log metrics
mlflow.end_run()  # End any active run
with mlflow.start_run():
    model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=1)
    
    # Evaluate model
    loss = model.evaluate(X_test, y_test, verbose=0)
    mlflow.log_metric("test_loss", loss)
    
    # Make predictions
    predictions = model.predict(X_test)
    
    # Log predictions
    mlflow.log_metric("prediction_mean", np.mean(predictions))
    mlflow.log_metric("prediction_std", np.std(predictions))

    # Inverse scaling
    predictions = scaler.inverse_transform(predictions)

    # Log prediction plot
    plt.figure()
    plt.plot(predictions, label='Predictions')
    plt.plot(y_test, label='Actual')
    plt.legend()
    plt.xlabel("Time")
    plt.ylabel("Sales")
    plt.title("Sales Prediction vs Actual")
    plt.savefig("predictions_plot.png")
    mlflow.log_artifact("predictions_plot.png")