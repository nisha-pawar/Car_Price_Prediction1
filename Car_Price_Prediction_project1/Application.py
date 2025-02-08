from flask import Flask, render_template, request
import pandas as pd
import pickle
import numpy as np

app = Flask(__name__)

model = pickle.load(open("Linear_Regression_Model.pkl", 'rb'))

car = pd.read_csv("Clining_Data.csv")


@app.route('/')
def index():
    names = sorted(car['Name'].unique())
    years = sorted(car['Year'].unique(), reverse=True)
    fuel_types = car['Fuel_Type'].unique()
    mileages = sorted(car['Mileage'].unique())
    engines = sorted(car['Engine'].unique())
    powers = sorted(car['Power'].unique())
    seats = sorted(car['Seats'].unique(), reverse=True)


    names.insert(0, "Select Name")

    return render_template('index.html', names=names, years=years, fuel_types=fuel_types, mileages=mileages, engines=engines, powers=powers, seats= seats)



@app.route('/predict', methods=['POST'])
def predict():
    name = request.form.get('name')
    year = int(request.form.get('year'))
    fuel_type = request.form.get('fuel_type')
    mileage = float(request.form.get('mileage'))
    engine = float(request.form.get('engine'))
    power = float(request.form.get('power'))
    seats = float(request.form.get('seats'))
    kilometers_driven=int(request.form.get('kilometers_driven'))

    # Print the received data for debugging
    print("Received data:", name, year, fuel_type, mileage, engine, power, seats, kilometers_driven)

    # Predicting the new price using the model
    prediction = model.predict(pd.DataFrame([[name, year, kilometers_driven, fuel_type, mileage, engine, power, seats]],
                                            columns=['Name','Year','Kilometers_Driven','Fuel_Type','Mileage','Engine','Power','Seats']))

    print(prediction)
    return str(np.round(prediction[0], 2))


if __name__ == "__main__":
    app.run(debug=True)
