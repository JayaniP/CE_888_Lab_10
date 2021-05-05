#!/usr/bin/env python
# coding: utf-8


# In[12]:


import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('D:\ModelDeployment\model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = round(prediction[0], 2)

    return render_template('D:\ModelDeployment\index.html', prediction_text='Employee Salary should be $ {}'.format(output))


# In[ ]:


if __name__ == "__main__":
    app.run(debug=True)
