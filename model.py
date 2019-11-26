# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 11:29:40 2019

@author: KAUSHIK OJHA
"""
# Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
import nltk
#nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
import pickle


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
# Importing the dataset
dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter = '\t', quoting = 3)


corpus = preprocess(dataset['Review'])
    
# Creating the Bag of Words model using CountVectorizer
cv = CountVectorizer(max_features = 2500)

X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, 1].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.19, random_state = 42)


#***************************************************
# Multinomial NB
print("Multinomial NB \n ")
# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import MultinomialNB
classifier = MultinomialNB(alpha=0.1)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)
#print(classifier.predict(xCheck))
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print ("Confusion Matrix:\n",cm)

# Accuracy, Precision and Recall
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
score1 = accuracy_score(y_test,y_pred)
score2 = precision_score(y_test,y_pred)
score3= recall_score(y_test,y_pred)
print("\n")
print("Accuracy is ",round(score1*100,2),"%")
print("Precision is ",round(score2,2))
print("Recall is ",round(score3,2))


#*****************************************************

# Saving model to disk
pickle.dump(classifier, open('model.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))
print(model.predict(X_test))


#Save vectorizer.vocabulary_
pickle.dump(cv.vocabulary_,open("feature.pkl","wb"))

#Load it later
loaded_vec = CountVectorizer(decode_error="replace",vocabulary=pickle.load(open("feature.pkl", "rb")))

