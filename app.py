from flask import Flask, render_template, request, redirect, url_for
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model
with open('model/life_expectancy_model.pkl', 'rb') as file:
    model = pickle.load(file)

@app.route('/')
def welcome():
    return render_template('welcome.html')

@app.route('/input', methods=['GET', 'POST'])
def input_data():
    if request.method == 'POST':
        # Store user info (name, email)
        user_info = {
            'name': request.form['name'],
            'email': request.form['email']
        }
        return redirect(url_for('predict'))
    return render_template('input.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Get input data from form
        year = float(request.form['year'])
        status = int(request.form['status'])
        adult_mortality = float(request.form['adult_mortality'])
        infant_deaths = float(request.form['infant_deaths'])
        alcohol = float(request.form['alcohol'])
        percentage_expenditure = float(request.form['percentage_expenditure'])
        hepatitis_b = float(request.form['hepatitis_b'])
        measles = float(request.form['measles'])
        bmi = float(request.form['bmi'])
        under_five_deaths = float(request.form['under_five_deaths'])
        polio = float(request.form['polio'])
        total_expenditure = float(request.form['total_expenditure'])
        diphtheria = float(request.form['diphtheria'])
        hiv_aids = float(request.form['hiv_aids'])
        gdp = float(request.form['gdp'])
        population = float(request.form['population'])
        thinness_1_19 = float(request.form['thinness_1_19'])
        thinness_5_9 = float(request.form['thinness_5_9'])
        income_composition = float(request.form['income_composition'])
        schooling = float(request.form['schooling'])

        # Prepare input for model
        input_data = np.array([[
            year, status, adult_mortality, infant_deaths, alcohol,
            percentage_expenditure, hepatitis_b, measles, bmi,
            under_five_deaths, polio, total_expenditure, diphtheria,
            hiv_aids, gdp, population, thinness_1_19, thinness_5_9,
            income_composition, schooling
        ]])
        
        # Make prediction
        prediction = model.predict(input_data)[0]
        
        return render_template('output.html', prediction=round(prediction, 2))
    
    return render_template('input.html')

if __name__ == '__main__':
    app.run(debug=True)