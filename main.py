from flask import Flask, request, jsonify, render_template
import pandas as pd
import pickle

app = Flask(__name__)
data=pd.read_csv('cleaned_data.csv')
pipe=pickle.load(open("ridge_model.pkl",'rb'))

@app.route('/')
def index():
    locations = sorted(data['location'].unique())
    area_types = ['Super built-up Area', 'Built-up Area', 'Plot Area', 'Carpet Area']
    return render_template('index.html', locations=locations, area_types=area_types)

@app.route('/predict', methods=['POST'])
def predict():

    location= request.form.get('location')
    bhk=request.form.get('bhk')
    sqft=request.form.get('sqft')
    bath=request.form.get('bath')
    area=request.form.get('area')

    print(location,bhk,bath,sqft,area)
    input=pd.DataFrame([[location,sqft,bath,area,bhk]],columns=['location','total_sqft','bath','area_type','bhk'])
    prediction=pipe.predict(input)[0] * 1e5


    return str(round(prediction,2))
if __name__ == "__main__":
    app.run(debug=True,port=5001)
