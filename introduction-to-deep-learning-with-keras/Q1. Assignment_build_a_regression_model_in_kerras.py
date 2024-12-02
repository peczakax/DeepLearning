# Peer-graded Assignment: Build a Regression Model in Keras


# Build a baseline model

# Use the Keras library to build a neural network with the following:
# - One hidden layer of 10 nodes, and a ReLU activation function
# - Use the adam optimizer and the mean squared error  as the loss function.

# 1. Randomly split the data into a training and test sets by holding 30% of the data for testing. You can use the 
# train_test_split helper function from Scikit-learn.

# 2. Train the model on the training data using 50 epochs.

# 3. Evaluate the model on the test data and compute the mean squared error between the predicted concrete strength and the actual concrete strength. You can use the mean_squared_error function from Scikit-learn.

# 4. Repeat steps 1 - 3, 50 times, i.e., create a list of 50 mean squared errors.

# 5. Report the mean and the standard deviation of the mean squared errors.

import pandas as pd
import numpy as np
import keras
import sklearn

from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from pprint import pprint

# Download and Clean Dataset from https://cocl.us/concrete_data
import requests
r = requests.get('https://cocl.us/concrete_data')
with open('concrete_data.csv', 'wb') as f:
    f.write(r.content)

import pandas as pd
concrete_data = pd.read_csv('concrete_data.csv')
concrete_data.head()

# check for missing values
concrete_data.isnull().sum()

# split data into predictors and target
concrete_data_columns = concrete_data.columns
predictors = concrete_data[concrete_data_columns[concrete_data_columns != 'Strength']] # all columns except Strength
target = concrete_data['Strength'] # Strength column

# define model
model = Sequential()
model.add(Dense(10, activation='relu', input_shape=(predictors.shape[1],)))
model.add(Dense(1))

# compile model
model.compile(optimizer='adam', loss='mean_squared_error')

# split data into training and test sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(predictors, target, test_size=0.3, random_state=42)

# train model
model.fit(X_train, y_train, epochs=50, verbose=2)

# evaluate model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print('Mean squared error:', mse)

# repeat experiment 50 times
mse_list = []
for i in range(50):
    X_train, X_test, y_train, y_test = train_test_split(predictors, target, test_size=0.3, random_state=i)
    model.fit(X_train, y_train, epochs=50, verbose=0)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mse_list.append(mse)

# print mean and standard deviation of mean squared errors
print('Mean of mean squared errors:', np.mean(mse_list))
print('Standard deviation of mean squared errors:', np.std(mse_list))
pprint(mse_list)