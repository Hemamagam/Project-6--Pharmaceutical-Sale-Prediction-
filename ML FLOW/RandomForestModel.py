import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Load train and test data
train_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")

# Assuming the target column is named "target"
X_train = train_data.drop(columns=['target'])
y_train = train_data['target']
X_test = test_data.drop(columns=['target'])
y_test = test_data['target']

def train_random_forest_model(X_train, y_train, X_test, y_test, n_estimators=100, max_depth=10):
    with mlflow.start_run():
        # Train model
        model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth)
        model.fit(X_train, y_train)

        # Make predictions
        y_pred = model.predict(X_test)

        # Evaluate model
        rmse = mean_squared_error(y_test, y_pred, squared=False)

        # Log parameters and metrics
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_metric("RMSE", rmse)

        # Log model
        mlflow.sklearn.log_model(model, "random_forest_model")

# Call the function with provided data
train_random_forest_model(X_train, y_train, X_test, y_test)