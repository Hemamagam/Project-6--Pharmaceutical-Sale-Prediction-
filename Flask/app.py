from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)

# Load the trained model
with open('random_forest_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Define a route for prediction
@app.route('/', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Get input parameters from the form
        store_id = int(request.form['Store_id'])
        date = request.form['Date']
        is_holiday = int(request.form['IsHoliday'])
        is_weekend = int(request.form['IsWeekend'])
        is_promo = int(request.form['IsPromo'])
        # Other parameters dependent on date
        # e.g., extract day, month, year from date
        # Example: (assuming date is in format 'YYYY-MM-DD')
        day = int(date.split('-')[2])
        month = int(date.split('-')[1])
        year = int(date.split('-')[0])

        # Prepare input data for prediction
        input_data = {
            'Store': store_id,
            'Customers': 0,  # Default value for Customers
            'Open': 1,  # Default value for Open
            'DayOfMonth': day,
            'Month': month,
            'Year': year,
            'Holiday': is_holiday,
            'IsWeekend': is_weekend,
            'Promo': is_promo
            # Add other parameters extracted from form if needed
        }
        input_df = pd.DataFrame(input_data, index=[0])

        # Make prediction
        prediction = model.predict(input_df)[0]

        # Render the results template with the prediction
        return render_template('results.html', prediction=prediction)

    # Render the index template for GET requests
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)