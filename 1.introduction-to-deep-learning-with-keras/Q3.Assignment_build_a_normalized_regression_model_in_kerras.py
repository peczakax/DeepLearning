# Peer-graded Assignment: Build a Regression Model in Keras

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

# Normalize the data is by subtracting the mean from the individual predictors and dividing by the standard deviation.
# Use a normalized version of the data.
#target_norm = (target - target.mean()) / target.std()
predictors = (predictors - predictors.mean()) / predictors.std()

# Use the Keras library to build a neural network with the following:
# - One hidden layer of 10 nodes, and a ReLU activation function
model = Sequential()
model.add(Dense(10, activation='relu', input_shape=(predictors.shape[1],)))
model.add(Dense(1))

# Use the adam optimizer and the mean squared error  as the loss function.
model.compile(optimizer='adam', loss='mean_squared_error')

# Randomly split the data into a training and test sets by holding 30% of the data for testing. Use the train_test_split helper function from Scikit-learn.
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(predictors, target, test_size=0.3, random_state=42)


# Train the model on the training data using 50 epochs.
model.fit(X_train, y_train, epochs=100, verbose=2)

# Evaluate the model on the test data and compute the mean squared error between the predicted concrete strength and the actual concrete strength. Use the mean_squared_error function from Scikit-learn.
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print('Mean squared error:', mse)

# Repeat steps 1 - 3, 50 times, i.e., create a list of 50 mean squared errors.
mse_list = []
for i in range(50):
    X_train, X_test, y_train, y_test = train_test_split(predictors, target, test_size=0.3, random_state=i)
    model.fit(X_train, y_train, epochs=50, verbose=0)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mse_list.append(mse)

# Report the mean and the standard deviation of the mean squared errors.
print('Mean of mean squared errors:', np.mean(mse_list))
print('Standard deviation of mean squared errors:', np.std(mse_list))
pprint(mse_list)
