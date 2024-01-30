from surprise import Dataset
from surprise.model_selection import train_test_split
from surprise import SVD
from surprise import accuracy
import numpy as np
import pandas as pd
from surprise import Reader
import time
import matplotlib.pyplot as plt
# Load the movielens-100k dataset
ratings = pd.read_csv('ml-latest-small/ratings.csv')
reader = Reader(rating_scale=(0.5, 5))
data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)

# Split into train and test sets
trainset, testset = train_test_split(data, random_state=42, test_size=0.2)

# Create an SVD model
svd = SVD()

# Start the timer
start_time = time.time()

# Fit the model on the train set
svd.fit(trainset)

# Stop the timer and calculate the training time
training_time = time.time() - start_time

# Make predictions for the test set
predictions = svd.test(testset)

# Compute RMSE and MAE
rmse = accuracy.rmse(predictions)
mae = accuracy.mae(predictions)

# Define a user id
user_id = 1

# Get the user ratings from the dataset
user_ratings = data.df[data.df['userId'] == user_id]

# Get the movie ids that the user has rated
rated_movies = user_ratings['movieId'].unique()

# Get all the movie ids from the dataset
all_movies = data.df['movieId'].unique()

# Get the movie ids that the user has not rated yet
unrated_movies = np.setdiff1d(all_movies, rated_movies)

# Create an empty list to store the predictions
predictions = []

# Loop over the unrated movies
for movie_id in rated_movies:
    # Predict the rating for each movie
    pred = svd.predict(user_id, movie_id)
    # Append the prediction to the list
    predictions.append(pred)

# Sort the predictions by their estimated ratings in descending order
predictions.sort(key=lambda x: x.est, reverse=True)

# Import the movies dataset
movies = pd.read_csv('ml-latest-small/movies.csv')

# Create a dictionary that maps movie ids to movie titles
movie_dict = dict(zip(movies['movieId'], movies['title']))

# Print the top 10 recommendations with the actual ratings
print('Top 10 recommendations for user', user_id, 'with actual ratings')
for pred in predictions[:20]:
    # Get the movie title from the dictionary
    movie_title = movie_dict[pred.iid]
    # Get the actual rating from the user ratings dataframe
    actual_rating = user_ratings[user_ratings['movieId'] == pred.iid]['rating'].values[0]
    # Print the movie title, estimated rating and actual rating
    print(f'MovieID: {pred.iid}, Estimated rating: {pred.est:.3f}, Actual rating: {actual_rating}')

# Print the training time
print('Training time:', training_time, 'seconds')


# Create two lists to store the actual and predicted ratings
actual_ratings = []
predicted_ratings = []

# Loop over the predictions
for pred in predictions:
    # Get the actual rating from the user ratings dataframe
    actual_rating = user_ratings[user_ratings['movieId'] == pred.iid]['rating'].values[0]
    # Get the predicted rating from the prediction object
    predicted_rating = pred.est
    # Append the ratings to the lists
    actual_ratings.append(actual_rating)
    predicted_ratings.append(predicted_rating)

plt.scatter(range(1, len(actual_ratings) + 1), actual_ratings, color='blue', label='Actual Ratings')
plt.plot(range(1, len(predicted_ratings) + 1), predicted_ratings, color='red', label='Predicted Values')
plt.xlabel('Data Point')
plt.ylabel('Rating')
plt.title('Actual Ratings vs Predicted Values for SVD')

plt.ylim((0, 5.5))
plt.xlim((0, 50.5))
plt.legend()
plt.show()
