#This is an example exercise for implementing the K- nearest neighbours. 
#The Dataset contains information about the marketing campaign on social networking for the products of SUV
#The dataset has variables such as Gender, Age, Salary and Purchased. Here the Purchased is the binary target variable and rest of them are independent variables. 
#Predictive model is being built using Age and Salary to predict whether the person buy the product or not.  
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Importing the dataset
Soc_Ads=pd.read_csv("Social_Network_Ads.csv")
X=Soc_Ads.iloc[:,[2,3]].values
Y=Soc_Ads.iloc[:,4].values
              
#Splittng the dataset into test and train
from sklearn.cross_validation import train_test_split
X_train,X_test,Y_train,Y_test= train_test_split(X,Y,test_size=0.2,random_state=10)

#Since the independent variables Age and Salary are continious, it is always good to scale them for accurate predictions
from sklearn.preprocessing import StandardScaler
sc_X=StandardScaler()
X_train=sc_X.fit_transform(X_train)
X_test=sc_X.fit_transform(X_test) 

#Building Model using K-Nearest Neighbours
#In this model building we are going to use the Euclidean distance as our Distance function to classify the cases
from sklearn.neighbors import KNeighborsClassifier
#In general, for most of the dataset, the K value ranges from 3-10. However, the optimal value can be derived based on how big the dataset is. 
#In this case, we are taking the K as 5 and using Eculidean distance as our metric
classifier=KNeighborsClassifier(5,metric='minkowski',p=2)
classifier.fit(X_train,Y_train)

#Predicting the test set results
y_pred=classifier.predict(X_test)

#Now that we have fitted our model, lets see how good our model has predicted by building the confusion matrix
from sklearn.metrics import confusion_matrix
ConfMat=confusion_matrix(Y_test,y_pred) 
ConfMat
#Lets look at the misclassfication rate and accuracy as our measurement criteria. 
#Misclassification rate is the Sum of False negative and False Postive divided by total observation in the test set.
#Accuracy is the sum of True Positive and True negative divided by the total observation in the test set.
MisClaRat=((ConfMat[0][1]+ConfMat[1][0])/len(Y_test))*100
MisClaRat
Accu = ((ConfMat[0][0]+ConfMat[1][1])/len(Y_test))*100   
Accu
#As we can see, our misclassification 7.5% and our accuracy is 92.5%
#Hence we can say that our model will predict 92.5% of the cases correctly. 

#Visualizing the results are always the fun part. Lets start with visualizing our training set first
from matplotlib.colors import ListedColormap
X,Y=X_train,Y_train
X1,X2=np.meshgrid(np.arange(start=X[:,0].min()-1,stop=X[:,0].max()+1,step=0.01),
                  np.arange(start=X[:,1].min()-1,stop=X[:,1].max()+1,step=0.01))
plt.contourf(X1,X2,classifier.predict(np.array([X1.ravel(),X2.ravel()]).T).reshape(X1.shape),alpha=0.75,cmap=ListedColormap(('red','green')))
plt.xlim(X1.min(),X1.max())
plt.ylim(X2.min(),X2.max())
for i,j in enumerate(np.unique(Y)):
    plt.scatter(X[Y==j,0],X[Y==j,1],
                c=ListedColormap(('red','green'))(i),label=j)
plt.title('KNN-training set')
plt.xlabel('Age')
plt.ylabel('Salary')
plt.legend()
plt.show()    
    
#Visualizing the test set
from matplotlib.colors import ListedColormap
X,Y=X_test,Y_test
X1,X2=np.meshgrid(np.arange(start=X[:,0].min()-1,stop=X[:,0].max()+1,step=0.01),
                  np.arange(start=X[:,1].min()-1,stop=X[:,1].max()+1,step=0.01))
plt.contourf(X1,X2,classifier.predict(np.array([X1.ravel(),X2.ravel()]).T).reshape(X1.shape),alpha=0.75,cmap=ListedColormap(('red','green')))
plt.xlim(X1.min(),X1.max())
plt.ylim(X2.min(),X2.max())
for i,j in enumerate(np.unique(Y)):
    plt.scatter(X[Y==j,0],X[Y==j,1],
                c=ListedColormap(('red','green'))(i),label=j)
plt.title('KNN-TestSet set')
plt.xlabel('Age')
plt.ylabel('Salary')
plt.legend()
plt.show()    

#we can see from the plot that the model has predicted the target variable perfectly and its a proper non linear classification

#Conclusion
#We can conclude that, the person with more in age are tending to buy the product and similarly the persons who are younger but having high salary are also tend to buy the product
          