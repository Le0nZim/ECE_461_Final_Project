# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 19:21:40 2024

@author: DaNi
"""

import matplotlib.pyplot as plt

# Elbow Method for finding the optimal number of clusters
inertia = []
K_range = range(1, 11)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_train)
    inertia.append(kmeans.inertia_)

# Plotting
plt.figure(figsize=(12, 6))
plt.plot(K_range, inertia, marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.title('Elbow Method For Optimal k')
plt.show()
