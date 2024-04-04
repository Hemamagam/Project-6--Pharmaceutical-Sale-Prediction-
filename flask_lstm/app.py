from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import pickle
import mlflow
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)

# Load models and other necessary data
with open('random_forest_model.pkl', 'rb') as sales_model_file:
    sales_model = pickle.load(sales_model_file)

with open('random_forest_model_customer.pkl', 'rb') as customers_model_file:
    customers_model = pickle.load(customers_model_file)

run = mlflow.active_run()
if run is not None:
    run_id = run.info.run_id
    lstm_model = mlflow.keras.load_model(f"runs:/{run_id}/my_model")
else:
    lstm_model = None
    # Handle the case when no active MLflow run is found


def forecast_sales(model, lstm_input, steps):
    forecast = []
    current_data = lstm_input.values[-model.input_shape[1]:].reshape((1, model.input_shape[1], lstm_input.shape[1]))
    for _ in range(steps):
        prediction = model.predict(current_data)
        forecast.append(prediction[0, 0])
        current_data = np.roll(current_data, -1)
        current_data[0, -1, 0] = prediction[0, 0]
    return forecast

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get input data from form
        input_data = {
            'Store': int(request.form['Store']),
            'Open': int(request.form['Open']),
            'Promo': int(request.form['Promo']),
            'Holiday': int(request.form['Holiday']),
            'DayOfMonth': int(request.form['DayOfMonth']),
            'Month': int(request.form['Month']),
            'Year': int(request.form['Year']),
            'IsWeekend': int(request.form['IsWeekend'])
        }

        # Prepare input data for prediction
        input_df = pd.DataFrame([input_data])

        # Make prediction for sales
        sales_prediction = sales_model.predict(input_df)

        # Make prediction for customers
        customers_prediction = customers_model.predict(input_df)

        if lstm_model is not None:
            # Forecast sales for next 42 days
            forecast_steps = 42
            lstm_input = input_df  # Define lstm_input
            forecast = forecast_sales(lstm_model, lstm_input, forecast_steps)

            # Plot forecast
            plt.figure(figsize=(12, 6))
            plt.plot(pd.date_range(start=input_df.index[-1], periods=forecast_steps, freq='D'), forecast, label='Forecasted Sales')
            plt.xlabel('Date')
            plt.ylabel('Sales')
            plt.title('Forecasted Sales for the Next 42 Days')
            plt.legend()
            forecast_path = 'static/forecast.png'  # Save the plot
            plt.savefig(forecast_path)  # Save the plot to a file
            plt.close()  # Close the plot to prevent memory leaks

        else:
            forecast_path = None

        # Create a DataFrame with the predictions
        prediction_df = pd.DataFrame({
            'Sales_Prediction': sales_prediction.flatten(),
            'Customers_Prediction': customers_prediction.flatten(), 
        })

        # Generate plot for sales vs customers
        plot_data = None
        if len(prediction_df) > 0:
            plt.figure(figsize=(5, 3))
            plt.scatter(prediction_df['Sales_Prediction'], prediction_df['Customers_Prediction'])
            plt.title('Sales vs Customers Prediction')
            plt.xlabel('Sales Prediction')
            plt.ylabel('Customers Prediction')
            plt.grid(True)
            plt.tight_layout()

            # Save plot to a bytes object
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png')
            buffer.seek(0)
            plot_data = base64.b64encode(buffer.getvalue()).decode()
            plt.close()  # Close the figure to release resources

        # Save predictions to CSV
        csv_path = 'static/predictions.csv'
        prediction_df.to_csv(csv_path, index=False)

        # Load CSV file into a DataFrame
        prediction_df = pd.read_csv(csv_path)

        return render_template('results.html', 
                               sales_prediction=sales_prediction[0],
                               customers_prediction=customers_prediction[0],  
                               forecast_plot=forecast_path, 
                               prediction_csv=prediction_df,
                               plot_data = plot_data)
                            
if __name__ == "__main__":
    if not os.path.exists('static'):
        os.makedirs('static')
    app.run(debug=True)