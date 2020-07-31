from flask import Flask,render_template,url_for,request
import pandas as pd
import numpy as np

import pickle

# load the model from disk
loaded_model=pickle.load(open('Random_forest_regressor.pkl', 'rb'))
app = Flask(__name__)

@app.route('/')
def home():
	return render_template('home.html')

@app.route('/predictcsv',methods=['POST'])
def predictcsv():
    df=pd.read_csv('Data/Real_Data/real_2018.csv')
    df.drop(labels = 'T',axis= 1,inplace = True)
    my_prediction=loaded_model.predict(df.iloc[:,:-1].values)
    my_prediction=my_prediction.tolist()
    return render_template('result.html',prediction = my_prediction)

@app.route('/predictexample',methods=['GET','POST'])
def predictexample():
    TM = request.form['TM']
    Tm = request.form['Tm']
    H = request.form['H']
    VV = request.form['VV']
    V = request.form['V']
    VM = request.form['VM']
    TM = int(TM)
    Tm = int(Tm)
    H = int(H)
    VV = int(VV)
    V = int(V)
    VM = int(VM)
    train = [TM,Tm,H,VV,V,VM]
    my_prediction=loaded_model.predict(np.array(train).reshape(1,-1))
    my_prediction=my_prediction.tolist()
    return render_template('result.html',prediction = my_prediction)

if __name__ == '__main__':
	app.run(port=5000,debug=True)
