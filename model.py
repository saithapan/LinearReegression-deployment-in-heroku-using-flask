# -*- coding: utf-8 -*-
"""
Created on Thu Dec 19 21:48:15 2019

@author: sai thapan ragipani
"""

#Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle

dataset = pd.read_csv("C:\\Users\\sai thapan ragipani\\flask deployment\\hiring.csv")

dataset['experience'].fillna(0,inplace = True)
dataset['test_score'].fillna(dataset['test_score'].mean(),inplace=True)
X = dataset.iloc[:,:3]

# converting word to integer values
def convert_to_int(word):
    word_dict = {'one':1,'two':2,'three':3,'four':4,'five':5,'six':6,'seven':7,'eight':8,
                'nine':9,'ten':10,'eleven':11,'twelve':12,'zero':0,0:0}
    
    return word_dict[word]

X['experience'] = X['experience'].apply(lambda x : convert_to_int(x))

y = dataset.iloc[:,-1]


# splitting training and test set

from sklearn.linear_model import LinearRegression

regressor = LinearRegression()

# fitting the  model with training data

regressor.fit(X,y)

# Saving model to disk
pickle.dump(regressor,open('model.pkl','wb'))


# loading the model to compare the results
with open('C:\\Users\\sai thapan ragipani\\flask deployment\\model.pkl', 'rb') as f:
    model=pickle.load(f)
    