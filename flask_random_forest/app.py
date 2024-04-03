from flask import Flask, render_template, request, send_file
import pandas as pd
import pickle
import os

app = Flask(__name__)

# Load the trained models for sales and customers
with open('random_forest_model.pkl', 'rb') as sales_model_file:
    sales_model = pickle.load(sales_model_file)

with open('random_forest_model_customer.pkl', 'rb') as customers_model_file:
    customers_model = pickle.load(customers_model_file)


# Define route for index page
@app.route('/')
def index():
    return render_template('index.html')

# Define route for prediction
@app.route('/predict', methods=['POST'])
def predict():
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

    # Render results template with predictions
    return render_template('results.html', sales_prediction=sales_prediction[0], customers_prediction=customers_prediction[0])

@app.route('/download', methods=['GET', 'POST'])
def download():
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

        input_df = pd.DataFrame([input_data])

        # Make prediction for sales
        sales_prediction = sales_model.predict(input_df)

        # Make prediction for customers
        customers_prediction = customers_model.predict(input_df)

        # Create a DataFrame with the predictions
        prediction_df = pd.DataFrame({
            'Sales_Prediction': sales_prediction,
            'Customer_Prediction': customers_prediction
        }) 

        # Specify the directory where the CSV file will be saved
        directory = os.path.join(r'C:/Users/pooji/AI Course Digicrome/One Python/Nexthike-Project Work/Project6Pharmaceutical/flask')

        # Save DataFrame to CSV with absolute file path
        file_path = os.path.join(directory, 'predictions.csv')
        prediction_df.to_csv(file_path, index=False)

        # Return the CSV file as a downloadable attachment
        return send_file(file_path, mimetype='text/csv', as_attachment=True)
    else:
        # Handle GET request
        return render_template('index.html')
    
if __name__ == "__main__":
    app.run(debug=True)
