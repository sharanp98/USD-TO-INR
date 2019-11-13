import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor 
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

from flask import Flask
from flask import request
from flask import render_template
app = Flask(__name__)


df = pd.read_csv('dataset.csv',engine='c',parse_dates=['DATE'])

df.DATE = df.DATE.astype(np.int64)

ls = []
def date_vs_features(date,feature):
    
    date = pd.to_datetime(pd.Series([date])).astype(np.int64)

    X = pd.DataFrame(df['DATE']).values
    y = pd.DataFrame(df[feature]).values


    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0)

    reg = RandomForestRegressor(n_estimators=10)
    
    y_train = y_train.ravel()

    reg.fit(X_train,y_train)
    
    temp = np.array([date]).reshape(1,-1)
    
    ls.append(reg.predict(temp)[0])
    
def predict_USD(date):
    
    col = list(df.columns)[2]
    date_vs_features(date,col)

    CPI = ls[0]
    
    date = pd.to_datetime(pd.Series([date])).astype(np.int64)

    X = df[['DATE','CPI']].values
    y = pd.DataFrame(df['USDTOINR']).values
    y = y.ravel()

    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

    reg = RandomForestRegressor(n_estimators = 100,random_state = 62)

    reg.fit(X_train,y_train)

    y_pred = reg.predict(X_test)
    
    USD = reg.predict (np.array([date,CPI]).reshape(1,-1))
    
    return "Prediction : 1 USD = " + str(round(USD[0],2)) + " INR"

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        return ( render_template('index.html',prediction_text = predict_USD(request.form['date']) ) )
    else:
        return render_template('index.html')