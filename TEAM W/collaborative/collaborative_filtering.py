# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 17:26:22 2024

@author: DaNi
"""

import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import numpy as np

# Load your data
cust_new = pd.read_csv('subset1.csv')

# Data Cleaning
cust_new.dropna(inplace=True)
cust_new = cust_new[cust_new['playscount'] > 0]

# Creating a pivot table
pivot_table = cust_new.pivot(index='CustID', columns='TrackId', values='playscount').fillna(0)

# Train-test split
train, test = train_test_split(pivot_table, test_size=0.25, random_state=42)

# Model Implementation - Using TruncatedSVD for matrix factorization
n_components = 90  # Number of latent factors. You can tune this parameter.
model = TruncatedSVD(n_components=n_components, random_state=42)
train_reduced = model.fit_transform(train)
test_reduced = model.transform(test)

# Predicting the missing entries in the user-item matrix
train_pred = np.dot(train_reduced, model.components_)
test_pred = np.dot(test_reduced, model.components_)

# Function to calculate RMSE
def rmse(true, pred):
    prediction = pred[true.nonzero()].flatten()
    truth = true[true.nonzero()].flatten()
    return np.sqrt(mean_squared_error(truth, prediction))

# Calculating error metrics
train_rmse = rmse(train.values, train_pred)
test_rmse = rmse(test.values, test_pred)

print(f"Train RMSE: {train_rmse}")
print(f"Test RMSE: {test_rmse}")

# Creating dictionaries to map IDs to indices
user_id_to_index = {id: idx for idx, id in enumerate(train.index)}
track_id_to_index = {id: idx for idx, id in enumerate(pivot_table.columns)}

def predict_playscount(user_id, track_id, user_matrix, item_matrix):
    # Check if user_id and track_id exist
    if user_id in user_id_to_index and track_id in track_id_to_index:
        # Get the indices for the user and the track
        user_idx = user_id_to_index[user_id]
        track_idx = track_id_to_index[track_id]

        # Retrieve user and item features
        user_features = user_matrix[user_idx]
        item_features = item_matrix[:, track_idx]

        # Predict playscount
        predicted_playscount = np.dot(user_features, item_features)
        return predicted_playscount
    else:
        return None  # or a default value

# Example prediction
user_id = 123  # Replace with a valid user ID
track_id = 1  # Replace with a valid track ID
predicted_playscount = predict_playscount(user_id, track_id, train_reduced, model.components_)

print(predicted_playscount)



