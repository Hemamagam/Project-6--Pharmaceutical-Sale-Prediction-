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
        input_data = {
            'Store': int(request.form['Store']),
            'Customers': int(request.form['Customers']),
            'Open': int(request.form['Open']),
            'Promo': int(request.form['Promo']),
            'Holiday': int(request.form['Holiday']),
            'DayOfMonth': int(request.form['DayOfMonth']),
            'Month': int(request.form['Month']),
            'Year': int(request.form['Year']),
            'IsWeekend': int(request.form['IsWeekend'])
        }

        # Prepare input data for prediction
        input_df = pd.DataFrame(input_data, index=[0])

        # Make prediction
        prediction = model.predict(input_df)[0]

        # Render the results template with the prediction
        return render_template('results.html', prediction=prediction)

    # Render the index template for GET requests
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)  
