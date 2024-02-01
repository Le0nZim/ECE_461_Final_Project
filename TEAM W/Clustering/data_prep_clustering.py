# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 18:35:29 2024

@author: DaNi
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
import random
from datetime import datetime, timedelta

# Load the datasets - Replace the file paths with your actual file paths
cust_df = pd.read_csv('cust.csv')
music_df = pd.read_csv('music.csv')
tracks = pd.read_csv('tracks.csv')

# User-Track Interaction Features
interaction_features = tracks.groupby('CustID').agg(
    total_plays=pd.NamedAgg(column='TrackId', aggfunc='count'),
    unique_tracks=pd.NamedAgg(column='TrackId', aggfunc=pd.Series.nunique)
).reset_index()

# Merge with Customer Demographic Data
merged_features = pd.merge(interaction_features, cust_df, on='CustID')
shape1 = merged_features.shape

#Convert 'SignDate' column to datetime format
merged_features['SignDate'] = pd.to_datetime(merged_features['SignDate'], format='%d/%m/%Y', errors='coerce')

# Find the minimum and maximum dates in the 'SignDate' column
min_date = merged_features['SignDate'].min()
max_date = merged_features['SignDate'].max()

# Function to generate random date within the specified range
def random_date(min_date, max_date):
    random_days = random.randint(0, (max_date - min_date).days)
    random_date = min_date + timedelta(days=random_days)
    return random_date.strftime('%d/%m/%Y')

# Function to check if a string is in 'DD/MM/YYYY' format
def is_valid_date(date_str):
    try:
        datetime.strptime(date_str, '%d/%m/%Y')
        return True
    except ValueError:
        return False

# Iterate through rows and fill rows with invalid date formats
for index, row in merged_features.iterrows():
    if not is_valid_date(str(row['SignDate'])):
        random_date_value = random_date(min_date, max_date)
        merged_features.at[index, 'SignDate'] = random_date_value

# Display the DataFrame with random date values for invalid formats
print(merged_features)


merged_features['SignDate'] = pd.to_datetime(merged_features['SignDate'], format='%d/%m/%y')

# Extract day, year, and month from 'SignDate'
merged_features['SignDate_Day'] = merged_features['SignDate'].dt.day.astype('category')
merged_features['SignDate_Year'] = merged_features['SignDate'].dt.year.astype('category')
merged_features['SignDate_Month'] = merged_features['SignDate'].dt.month.astype('category')

merged_features['zip'] = merged_features['zip'].astype('category')

merged_features.drop(['Address', 'Name', 'SignDate', 'LinkedWithApps', "Campaign", "Status"], axis=1, inplace=True)

# Display the DataFrame with categorical day, year, and month columns
print(merged_features.head())



# Encoding Categorical Data
label_encoder = LabelEncoder()
for col in ['Gender', 'Level', 'zip', 'SignDate_Day', 'SignDate_Year', 'SignDate_Month']:
    merged_features[col] = label_encoder.fit_transform(merged_features[col])

# Normalize Data
scaler = MinMaxScaler()
merged_features[['total_plays', 'unique_tracks']] = scaler.fit_transform(
    merged_features[['total_plays', 'unique_tracks']]
)


