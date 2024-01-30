Machine Learning Final Project 2023-24

TEAM G:
- Kavvathas Spyridon (02928), skavvathas@uth.gr
- Lagomatis Ilias    (03005), ilagomatis@uth.gr
- Petridis Christos  (03086), cpetridis@uth.gr

There are some notebooks which contain the code of different recommendation system implementations. Below is a brief description of each file and its role in the project.

**** TF-IDF.ipynb : This notebook explores the implementation of Term Frequency-Inverse Document Frequency (TF-IDF) for the content-based recommendation system.

**** Jaccard-Similarity.ipynb: This notebook explores the implementation of the Jaccard Similarity method.

**** Matrix-Factorization.ipynb: This notebook explores the implementation of some matrix factorization techniques such as SVM and NMF (collaborative filtering).

**** kmeans.ipynb: This notebook implements the k-means clustering algorithm, which is an unsupervised learning algorithm used to group similar courses into clusters (content-based).

**** kNN-users.ipynb: This notebook implements the k-Nearest Neighbors (kNN) algorithm. This method involves finding users that are similar to the target user and recommending items that these similar users have liked or interacted with.

**** kNN-courses.ipynb: This notebook focuses on item-based collaborative filtering using the kNN algorithm. Instead of finding similar users, the algorithm finds similar items based on user interactions and recommends these items to the user.

**** content-based neural net.ipynb: This notebook details the implementation of a content-based recommendation system using neural networks (Autoencoder & Embeddings).



Datasets:
- rating_coursera_final.csv contains the ratings/interactions between users and courses on Coursera_courses.csv (ratings 1-5)
- Coursera_courses.csv contains the information for each course (title, institution etc.) that users have rated.
- Coursera.csv contains more information for each course (title, institution, description, difficulty level etc.) and this dataset is being used for content-based algorithms.


User interface
We have also implemented a user-interface that one can easily access it and see the recommendations for a given input. However, some very large files are required (such as tf_idf_matrix and svd_matrix), There are more details and examples in the deliverable report.