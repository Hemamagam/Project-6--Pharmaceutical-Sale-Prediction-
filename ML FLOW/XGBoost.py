import mlflow
import mlflow.xgboost
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# Assuming you have train_data_processed.csv
train_data = pd.read_csv("train_data_processed.csv")

train_data['Date'] = pd.to_datetime(train_data['Date'])

# Extract features from Date column
train_data['Year'] = train_data['Date'].dt.year
train_data['Month'] = train_data['Date'].dt.month
train_data['Day'] = train_data['Date'].dt.day

X_train = train_data.drop(columns=['Sales', 'Date'])  # Drop 'Sales' and 'Date' columns
y_train = train_data['Sales']

def train_xgboost_model(X_train, y_train, params):
    with mlflow.start_run():
        # Train model
        dtrain = xgb.DMatrix(X_train, label=y_train)
        model = xgb.train(params, dtrain)

        # Save model locally in JSON format
        model_path = "xgboost_model.json"
        model.save_model(model_path)

        # Log parameters and metrics
        mlflow.log_params(params)

        # Log model
        mlflow.xgboost.load_model(model_path, "xgboost_model")

# Define XGBoost parameters
xgboost_params = {
    'objective': 'reg:squarederror',
    'eval_metric': 'rmse',
    'max_depth': 6,
    'learning_rate': 0.1,
    'n_estimators': 100,
    'enable_categorical': True  # Enable categorical feature handling
}

# Call the function with provided data and parameters
train_xgboost_model(X_train, y_train, xgboost_params)