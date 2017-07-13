# -*- coding: utf-8 -*-
"""
Created on Fri May 26 01:14:21 2017

@author: Kasthuri Rengan
"""
#This dataset is about 50 Startups which are in the State of Newyork, California and Florida. 
#The Dataset consists of 5 Variables R&D Spend, Administration, Marketing Spend, State(Newyork, California and Florida) and Profit
#The goal of this exercise is to model the data and analyze what factors determine the profit ( Dependent Variable).


#Importing Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as sm
#Importing the dataset
dataset = pd.read_csv("50_Startups.csv")
X = dataset.iloc[:,:-1].values
Y = dataset.iloc[:,4].values
                
# Encoding categorical data
# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 3] = labelencoder_X.fit_transform(X[:, 3])
onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()

#Avoiding the dummy variable trap
X=X[:,1:]

#Splitting the dataset.
from sklearn.cross_validation import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=0)

#Fitting the multiple Linear Regression Model.
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,Y_train)#Fitting the regressor to our training set. 

#Predicting the test results. 
y_pred = regressor.predict(X_test)

#Eventhough this model has predicted the results in a very good way, we can still go one step ahead in tuning our model using Backward Elimination
X=np.append(arr=np.ones((50,1)).astype(int),values=X,axis=1)
#we are doing that becasue, we need to add an 1 to the intercept so that it wont get confused. Also we need to put those 50 ones in the first in arr because we need to add that in the begining of the array. 
#Ones is the place, where we can add an value. The first number corresponds to the number of observations and one column. 

X_OPT=X[:,[0,1,2,3,4,5]]
regressor_OLS=sm.OLS(endog=Y,exog=X_OPT).fit()
regressor_OLS.summary()
#As we can see from the summary, we can see that the x2 is having higher p value, considering the significance level of 0.05

X_OPT=X[:,[0,1,3,4,5]]
regressor_OLS=sm.OLS(endog=Y,exog=X_OPT).fit()
regressor_OLS.summary()

X_OPT=X[:,[0,3,4,5]]
regressor_OLS=sm.OLS(endog=Y,exog=X_OPT).fit()
regressor_OLS.summary()

X_OPT=X[:,[0,3,5]]
regressor_OLS=sm.OLS(endog=Y,exog=X_OPT).fit()
regressor_OLS.summary()

X_OPT=X[:,[0,3]]
regressor_OLS=sm.OLS(endog=Y,exog=X_OPT).fit()
regressor_OLS.summary()

#Finally we can conclude that, only the Variable R&D Spend is more significant than the other variables in determining the profit. 
#So as a investor, one has to evaluate the company's R&D Spend to get more profit from the investment they make. 