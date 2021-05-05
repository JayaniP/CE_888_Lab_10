#!/usr/bin/env python
# coding: utf-8

# In[2]:


pip install flask


# In[3]:


import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle


# In[6]:


app = Flask(__name__)
model = pickle.load(open('D:\CE888_Data Science_Decision Making\Model_Deployment\model.pkl', 'rb'))


# In[7]:


@app.route('/')
def home():
    return render_template('D:\CE888_Data Science_Decision Making\Model_Deployment\index.html')


# In[9]:


@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = round(prediction[0], 2)

    return render_template('D:\CE888_Data Science_Decision Making\Model_Deployment\index.html', prediction_text='Employee Salary should be $ {}'.format(output))


# In[ ]:


if __name__ == "__main__":
    app.run(debug=True)

