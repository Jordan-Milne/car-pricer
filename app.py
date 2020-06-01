from flask import Flask, render_template, request, jsonify
import dill
import pandas as pd
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv('cars1.csv')
df = df[df.price > 100]
df = df[df.year > 2000]
df = df[df.odometer < 1000000]
df['name'] = df['manufacturer'].str.title() + ' ' + df['model'].str.title()
df['name']

le = LabelEncoder()
le.fit(df['name'])
df['model'] = le.transform(df['name'])






app = Flask(__name__, template_folder='templates')
#-------- MODEL GOES HERE -----------#

pipe = dill.load(open("pipe.pkl", 'rb'))

#-------- ROUTES GO HERE -----------#

@app.route('/')
def index():
    return render_template('indexx.html')

@app.route('/result', methods=['POST'])
def predict_price(): 
    args = request.form
    print(args)
    data = pd.DataFrame({
        'year': [args.get('year')],
        'model': [le.transform([args.get('name')])[0]],
        'cylinders': [args.get('cylinders')],
        'fuel': [args.get('fuel')],
        'odometer': [args.get('odometer')],
        'transmission': [args.get('transmission')],
        'paint_color': [args.get('paint_color')],
        'condition': [args.get('condition')],
    })

    prediction = f'${round(float(pipe.predict(data)[0]*1.4),-1)} CAD'
    predd = round(float(pipe.predict(data)[0]),-1)

    return render_template('result.html', prediction=prediction)


if __name__ == '__main__':
    app.run(port=5000, debug=True)
