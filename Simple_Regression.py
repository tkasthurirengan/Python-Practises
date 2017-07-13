#This is a pretty simple dataset to demonstrate the linear regression. This dataset consists of two variables experience and Salaray. 
#This dataset will be used by HR for negotiation of the potential employees. 
#The independent variable is Years Of Experience and Dependent Variable is Salary.

#Importing Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv("Salary_Data.csv")
X = dataset.iloc[:,:-1].values
Y = dataset.iloc[:,1].values
                
#Splitting the dataset. 
from sklearn.cross_validation import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=1/3,random_state=0)

#Fitting simple linear regression in training set. 
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,Y_train)

#Predicting the results

y_pred = regressor.predict(X_test)

#Visualizing the test results
plt.scatter(X_train,Y_train,color='Red')
plt.plot(X_train,regressor.predict(X_train))
plt.title("Salary Vs Experience")
plt.show()

#Plotting the Results

plt.scatter(X_test,Y_test,color='Red')
plt.plot(X_train,regressor.predict(X_train))
plt.title("Salary Vs Experience")
plt.show()
