# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 18:14:20 2024

@author: DaNi
"""

from sklearn.metrics import mean_squared_error
from sklearn.decomposition import TruncatedSVD
import numpy as np
from sklearn.model_selection import train_test_split

# Load your data
cust_new = pd.read_csv('subset1.csv')

# Data Cleaning
cust_new.dropna(inplace=True)
cust_new = cust_new[cust_new['playscount'] > 0]

# Creating a pivot table
pivot_table = cust_new.pivot(index='CustID', columns='TrackId', values='playscount').fillna(0)

# Train-test split
train, test = train_test_split(pivot_table, test_size=0.25, random_state=42)

# Hyperparameter tuning for n_components
components_range = range(10, 100, 10)  # Example range (10, 20, ..., 90)
best_rmse = float('inf')
best_n_components = 0

for n_components in components_range:
    model = TruncatedSVD(n_components=n_components, random_state=42)
    train_reduced = model.fit_transform(train)
    test_reduced = model.transform(test)

    # Predicting the missing entries in the user-item matrix
    train_pred = np.dot(train_reduced, model.components_)
    test_pred = np.dot(test_reduced, model.components_)

    # RMSE calculation
    def rmse(true, pred):
        prediction = pred[true.nonzero()].flatten()
        truth = true[true.nonzero()].flatten()
        return np.sqrt(mean_squared_error(truth, prediction))

    # Calculating error metrics
    test_rmse = rmse(test.values, test_pred)

    # Update best RMSE and n_components
    if test_rmse < best_rmse:
        best_rmse = test_rmse
        best_n_components = n_components

print(f"Best RMSE: {best_rmse} for n_components: {best_n_components}")
