from flask import Flask, render_template, request, jsonify
import pickle
import pandas as pd

app = Flask(__name__, template_folder='templates')
#-------- MODEL GOES HERE -----------#

pipe = pickle.load(open("pipe.pkl", 'rb'))

#-------- ROUTES GO HERE -----------#

@app.route('/')
def index():
    return render_template('indexx.html')

@app.route('/result', methods=['POST'])
def predict_price():
    args = request.form
    print(args)
    data = pd.DataFrame({
        'Name': [args.get('Name')],
        'Location': [args.get('Location')],
        'Year': [args.get('Year')],
        'Kilometers_Driven': [args.get('Kilometers_Driven')],
        'Fuel_Type': [args.get('Fuel_Type')],
        'Transmission': [args.get('Transmission')],
        'Owner_Type': [args.get('Owner_Type')],
        'Mileage': [args.get('Mileage')],
        'Engine': [args.get('Engine')],
        'Power': [args.get('Power')],
        'Seats': [args.get('Seats')],
    })
    prediction = f'${round(float(pipe.predict(data)[0]),-1)} CAD'
    predd = round(float(pipe.predict(data)[0]),-1)

    return render_template('result.html', prediction=prediction) if (predd < 999999) and (predd > 0) else render_template('result.html', prediction="You have selected the wrong fuel type or transmission type for that model of car OR you have entered an unrealistic value")


if __name__ == '__main__':
    app.run(port=5000, debug=True)
