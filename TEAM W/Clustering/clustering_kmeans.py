# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 19:30:44 2024

@author: DaNi
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from collections import Counter

# ... [Your initial data loading and processing steps here] ...

# Split the data for training and testing
X_train, X_test = train_test_split(merged_features, test_size=0.2, random_state=42)

# K-Means Clustering
kmeans = KMeans(n_clusters=20, random_state=42)
kmeans.fit(X_train)

# Predicting clusters for the test set
test_clusters = kmeans.predict(X_test)

# Evaluating Clustering
silhouette_score_value = silhouette_score(X_test, test_clusters)
print(f"Silhouette Score: {silhouette_score_value}")

# Predicting clusters for merged_features
merged_features['Cluster'] = kmeans.predict(merged_features)

# Assign clusters to tracks based on customer data
tracks_with_clusters = tracks.merge(merged_features[['CustID', 'Cluster']], on='CustID', how='left')

# Group by Cluster and TrackId in the tracks data, and count occurrences
cluster_track_counts = tracks_with_clusters.groupby(['Cluster', 'TrackId']).size().reset_index(name='Count')

# Sort within each cluster by count to find the most popular tracks
cluster_track_counts.sort_values(['Cluster', 'Count'], ascending=[True, False], inplace=True)

# Print the top 10 tracks for each cluster
for cluster in range(20):  # Assuming 10 clusters
    top_tracks_ids = cluster_track_counts[cluster_track_counts['Cluster'] == cluster][:10]['TrackId']
    top_tracks = music_df[music_df['TrackId'].isin(top_tracks_ids)]
    print(f"\nTop 10 Tracks for Cluster {cluster}:")
    print(top_tracks[['TrackId', 'Title']].to_string(index=False))


# Print Customer ID and their Cluster
for cust_id, cluster in zip(merged_features['CustID'], merged_features['Cluster']):
    print(f"Customer ID: {cust_id}, Cluster: {cluster}")



