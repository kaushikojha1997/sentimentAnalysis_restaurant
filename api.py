# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 12:14:37 2019

@author: KAUSHIK OJHA
"""
import re

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

from flask import Flask, request, jsonify, render_template,g
import pickle
import flask_cors
from flask_cors import CORS
from sklearn.feature_extraction.text import CountVectorizer
import csv
from datetime import datetime

print(flask_cors.__version__)
app = Flask(__name__)

cors = CORS(app, resources={r"/api/*": {"origins": "*"}})

model = pickle.load(open('model.pkl', 'rb'))
cv = CountVectorizer(decode_error="replace",vocabulary=pickle.load(open("feature.pkl", "rb")))

@app.before_request
def before_request_func():
    
    g.csvPos = open("positiveReviews.csv", 'a')
    g.csvWriterPos = csv.writer(g.csvPos)
    g.csvNeg = open("negativeReviews.csv", 'a')
    g.csvWriterNeg = csv.writer(g.csvNeg)
    

def preprocess(dataset):
    corpus=[]
    for i in range(len(dataset)):
        review = re.sub('[^a-zA-Z]', ' ', dataset[i])
        review = review.lower()
        review = review.split()
        ps = PorterStemmer()
        review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
        review = ' '.join(review)
        corpus.append(review)
    return corpus

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    name = request.form.get('name')
    dept = request.form.get('department')
    review = request.form.get('review')
    x = []
    #json_ = jsonify({'Review': review})
    #pd.DataFrame(json_)
    x.append(review)
    print(review)
    
    corpus = preprocess(x)
    
    X = cv.fit_transform(corpus).toarray()
    prediction = model.predict(X)
    if prediction[0]==1:
        out="Positive"
        g.csvWriterPos.writerow([time,name,dept,review])
        #csvPos.close()
        return render_template('index.html', prediction_text='Thank You, hope you visit us again: {}'.format(out))
    else:
        out="Negative"    
        g.csvWriterNeg.writerow([time,name,dept,review])
        #csvNeg.close()
        return render_template('index.html', prediction_text='We will definetly look into this, and make ourself better: {}'.format(out))
    
    return render_template('index.html')
@app.route('/predict_api',methods=['POST'])
def predict_api():
    '''
    For direct API calls trought request
    
    '''

    review = request.form.get('review')
    x = []
    #json_ = jsonify({'Review': review})
    #pd.DataFrame(json_)
    x.append(review)
    print(review)
    
    corpus = preprocess(x)
    
    X = cv.fit_transform(corpus).toarray()
    prediction = model.predict(X)
    if prediction[0]==1:
        out="Positive"
    else:
        out="Negative"
    output = jsonify({'output':out})
    return (output)


@app.teardown_request
def teardown_request_func(error=None):
    g.csvPos.close()
    g.csvNeg.close()

if __name__ == '__main__':
    try:
        port = int(8880) # This is for a command-line input
    except:
        port = 12345 # If you don't provide any port the port will be set to 12345


    app.run(debug=True)