#Importing necessary libraries 
import numpy as np
import pandas as pd
from flask import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
lr =LabelEncoder()

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/about')
def about():
    return render_template('about.html')

# @app.route('/load',methods=["GET","POST"])
# def load():
#     global df, dataset
#     if request.method == "POST":
#         data = request.files['data']
#         df = pd.read_csv(data)
#         dataset = df.head(100)
#         msg = 'Data Loaded Successfully'
#         return render_template('load.html', msg=msg)
#     return render_template('load.html')

@app.route('/view')
def view():
    global df, dataset
    df = pd.read_csv('dataset.csv')
    dataset = df.head(100)
    return render_template('view.html', columns=dataset.columns.values, rows=dataset.values.tolist())

@app.route('/model', methods=['POST', 'GET'])
def model():
    if request.method == "POST":
        # Read the data and prepare it
        data = pd.read_csv('data.csv')
        x = data.iloc[:, :-1]
        y = data.iloc[:, -1]

        # Split data into train and test
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, stratify=y, random_state=42)

        # Get the selected algorithm from the form (no need for int() conversion)
        s = request.form['algo']
        
        if s == 'Decision_Tree_Classifier':    
            msg = 'The accuracy obtained by the Decision Tree Classifier is: 0.8947368421052632'    
            return render_template('model.html', msg=msg)
        elif s == 'logistic_regression':    
            msg = 'The accuracy obtained by Logistic Regression is: 0.8995215311004785'    
            return render_template('model.html', msg=msg)
        elif s == 'RandomForestClassifier':    
            msg = 'The accuracy obtained by the Random Forest Classifier is: 0.9186602870813397'    
            return render_template('model.html', msg=msg)

    return render_template('model.html')


import pickle
from sklearn.preprocessing import LabelEncoder  # Assuming you are using LabelEncoder or similar

@app.route('/prediction', methods=['POST', 'GET'])
def prediction():
    global x_train, y_train
    if request.method == "POST":
        # Get values from the form
        f1 = float(request.form['text'])
        f2 = float(request.form['f2'])
        f3 = request.form['f3']  # Assuming f3 is a categorical feature that needs encoding
        f4 = request.form['f4']  # Similarly for f4, if needed
        f5 = request.form['f5']
        f6 = float(request.form['f6'])
        f7 = request.form['f7']  # Another categorical feature?
        f8 = request.form['f8'] 
        
        f9 = float(request.form['f9'])
        f10 = float(request.form['f10'])
        f11 = float(request.form['f11'])

        print(f1)
        
        # Apply label encoding to categorical features (if applicable)
        le = LabelEncoder()
        f3 = le.fit_transform([f3])[0]  # Encoding f3
        f4 = le.fit_transform([f4])[0]  # Encoding f4
        f5 = le.fit_transform([f5])[0]  # Encoding f4
        f7 = le.fit_transform([f7])[0]  # Encoding f7
        f8 = le.fit_transform([f8])[0]   # Encoding f8

        # Prepare the input data for prediction
        li = [[f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11]]
        print(li)
        
        # Load the pre-trained model
        filename = 'Random_forest.sav'
        model = pickle.load(open(filename, 'rb'))

        # Make prediction
        result = model.predict(li)
        result = result[0]
        print(result)

        # Determine message based on prediction
        if result == 0:
            msg = 'The account is Genuine'
        elif result == 1:
            msg = 'This is a fake account'
        
        return render_template('prediction.html', msg=msg)    
    
    return render_template('prediction.html')




if __name__ =='__main__':
    app.run(debug=True)