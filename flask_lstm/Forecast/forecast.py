from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import pickle
import os
import matplotlib
matplotlib.use('Agg')  # Use 'Agg' backend to avoid threading issues
import matplotlib.pyplot as plt
import io
import base64
from datetime import datetime, timedelta

app = Flask(__name__)

# Load models and other necessary data
with open('random_forest_model.pkl', 'rb') as sales_model_file:
    sales_model = pickle.load(sales_model_file)

with open('random_forest_model_customer.pkl', 'rb') as customers_model_file:
    customers_model = pickle.load(customers_model_file)

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

        # Create a date object using DayOfMonth, Month, and Year
        start_date = datetime(input_data['Year'], input_data['Month'], input_data['DayOfMonth'])

        # Prepare input data for prediction
        input_df = pd.DataFrame([input_data])

        # Generate predictions for the next 6 weeks
        predictions = []
        for i in range(6 * 7):  # 6 weeks * 7 days
            prediction_date = start_date + timedelta(days=i)
            input_df_copy = input_df.copy()  # Make a copy to avoid modifying original input_df
            input_df_copy['DayOfMonth'] = prediction_date.day
            input_df_copy['Month'] = prediction_date.month
            input_df_copy['Year'] = prediction_date.year
            input_df_copy['IsWeekend'] = 1 if prediction_date.weekday() >= 5 else 0

            # Make prediction for sales
            sales_prediction = sales_model.predict(input_df_copy)

            # Make prediction for customers
            customers_prediction = customers_model.predict(input_df_copy)

            predictions.append({
                'Date': prediction_date.strftime('%Y-%m-%d'),
                'Sales_Prediction': sales_prediction[0],
                'Customers_Prediction': customers_prediction[0]
            })

        # Create a DataFrame with the predictions
        prediction_df = pd.DataFrame(predictions)

        # Generate plot for sales vs customers
        plot_data = None
        if len(prediction_df) > 0:
            plt.figure(figsize=(10, 6))
            plt.plot(prediction_df['Date'], prediction_df['Sales_Prediction'], label='Sales Prediction')
            plt.plot(prediction_df['Date'], prediction_df['Customers_Prediction'], label='Customers Prediction')
            plt.title('Sales and Customers Forecast')
            plt.xlabel('Date')
            plt.ylabel('Forecast')
            plt.xticks(rotation=45)
            plt.legend()
            plt.tight_layout()

            # Save plot to a bytes object
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png')
            buffer.seek(0)
            plot_data = base64.b64encode(buffer.getvalue()).decode()
            plt.close()  # Close the figure to release resources

        # Save predictions to CSV
        csv_path = 'static/predictions.csv'
        prediction_df.to_csv(csv_path, index=False)

        return render_template('results.html', 
                               prediction_csv=prediction_df,
                               plot_data=plot_data)

if __name__ == "__main__":
    if not os.path.exists('static'):
        os.makedirs('static')
    app.run(debug=True, port=5003)
