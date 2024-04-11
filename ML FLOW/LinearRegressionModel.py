import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# Assuming you have train.csv and test.csv in the same directory

# Load data
train_data = pd.read_csv("train_data_processed.csv")
test_data = pd.read_csv("test.csv")

# Split data into features and target variable
X_train = train_data.drop(columns=['target_column'])
y_train = train_data['target_column']
X_test = test_data.drop(columns=['target_column'])
y_test = test_data['target_column']

def train_linear_regression_model(X_train, y_train, X_test, y_test):
    with mlflow.start_run():
        # Train model
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Make predictions
        y_pred = model.predict(X_test)

        # Evaluate model
        rmse = mean_squared_error(y_test, y_pred, squared=False)

        # Log metrics
        mlflow.log_metric("RMSE", rmse)

        # Log model
        mlflow.sklearn.log_model(model, "linear_regression_model")

# Call the function
train_linear_regression_model(X_train, y_train, X_test, y_test)