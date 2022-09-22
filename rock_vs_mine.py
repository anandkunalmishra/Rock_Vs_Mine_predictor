
"""
Importing the Dependencies
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

"""Data Collection and Data Processing """

#loading datas to the pandas dataframe.
sonar_data = pd.read_csv('/sonar_data.csv',header=None)

sonar_data.head()

# number of rows and coloumn
sonar_data.shape

sonar_data.describe()

sonar_data[60].value_counts()

"""M --> Mine

R --> Rock
"""

sonar_data.groupby(60).mean()

#seperating data and labels
X = sonar_data.drop(columns=60,axis=1)
Y = sonar_data[60]

print(X)
print(Y)

"""Training and Test data"""

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.1,stratify=Y,random_state=1)

print(X.shape,X_train.shape,X_test.shape)

print(X_train)
print(Y_train)

"""Model Training --> Logistic Regression()"""

model = LogisticRegression()

#training the Logistic Regression Model with training data
model.fit(X_train,Y_train)

"""Model Evaluation"""

#accuracy on training data
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction,Y_train)

print('Accuracy on training data :',training_data_accuracy)

#accuracy on training data
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction,Y_test)

print('Accuracy on test data :',test_data_accuracy)

"""Making a Predictive System"""

input_data = (0.0454,0.0472,0.0697,0.1021,0.1397,0.1493,0.1487,0.0771,0.1171,0.1675,0.2799,0.3323,0.4012,0.4296,0.5350,0.5411,0.6870,0.8045,0.9194,0.9169,1.0000,0.9972,0.9093,0.7918,0.6705,0.5324,0.3572,0.2484,0.3161,0.3775,0.3138,0.1713,0.2937,0.5234,0.5926,0.5437,0.4516,0.3379,0.3215,0.2178,0.1674,0.2634,0.2980,0.2037,0.1155,0.0919,0.0882,0.0228,0.0380,0.0142,0.0137,0.0120,0.0042,0.0238,0.0129,0.0084,0.0218,0.0321,0.0154,0.0053)
#changing the input_data into numpy array
input_data_as_numpy_array = np.asarray(input_data)
#reshape the np array as we are predicting for one instance 
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction = model.predict(input_data_reshaped)
print(prediction)
if(prediction[0]=='R'):print("Object is a Rock")
else : print("Object is a Mine")
