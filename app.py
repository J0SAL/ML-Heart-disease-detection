from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd

model_path = 'models/knn.pkl'
model = pickle.load(open(model_path,'rb'))

scaler_path = 'models/standardScaler.pkl'
scaler = pickle.load(open(scaler_path,'rb'))

app = Flask(__name__)


@app.route('/')
def Home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():

    name     = request.form['name']
    age      = int(request.form['age'])
    sex      = int(request.form['sex'])
    cp       = int(request.form['cp'])    
    trestbps = int(request.form['trestbps'])
    chol     = int(request.form['chol'])
    fbs      = int(request.form['fbs'])
    restecg  = int(request.form['restecg'])
    thalach  = int(request.form['thalach'])
    exang    = int(request.form['exang'])
    oldpeak  = int(request.form['oldpeak'])
    slope    = int(request.form['slope'])
    ca       = int(request.form['ca'])
    thal     = int(request.form['thal'])

    #feature scaling
    temp     = [age, trestbps, chol, thalach, oldpeak]
    temp     = scaler.transform([temp])
    temp     = list(temp[0])

    #changing categorical to dummy data
    sex     = list(pd.get_dummies(list(range(0,2)))[sex])
    cp      = list(pd.get_dummies(list(range(0,4)))[cp])
    fbs     = list(pd.get_dummies(list(range(0,2)))[fbs])
    restecg = list(pd.get_dummies(list(range(0,3)))[restecg])
    exang   = list(pd.get_dummies(list(range(0,2)))[exang])
    slope   = list(pd.get_dummies(list(range(0,3)))[slope])
    ca      = list(pd.get_dummies(list(range(0,5)))[ca])
    thal    = list(pd.get_dummies(list(range(0,4)))[thal])

    temp.extend(sex+cp+fbs+restecg+exang+slope+ca+thal)

    arr     = np.array([temp])
    pred    = model.predict(arr)

    return render_template('index.html', data=pred, name=name)


if __name__ == "__main__":
    app.run(debug=False)

