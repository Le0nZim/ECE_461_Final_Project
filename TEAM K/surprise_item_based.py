# Import libraries
import pandas as pd
import matplotlib.pyplot as plt
from surprise import Reader, Dataset, KNNWithMeans, accuracy
from surprise.model_selection import train_test_split
import time

# Load the dataset
ratings = pd.read_csv('ml-latest-small/ratings.csv')
movies = pd.read_csv('ml-latest-small/movies.csv')

# Merging movies with ratings
df = pd.merge(movies, ratings, on='movieId')

reader = Reader(rating_scale=(0.5, 5))
data = Dataset.load_from_df(df[['userId', 'movieId', 'rating']], reader)
trainset, testset = train_test_split(data, random_state=42, test_size=0.2)

# User-based collaborative filtering
model = KNNWithMeans(sim_options={'name': 'cosine', 'user_based': False})

start_time = time.time()
model.fit(trainset)
training_time = time.time() - start_time

# Test the model and compute RMSE
predictions = model.test(testset)
rmse = accuracy.rmse(predictions)

# Get the list of all movie ids
all_movie_ids = df['movieId'].unique()

# Get a list of movie ids that user 1 has rated
rated_movies = df[df['userId'] == 1]['movieId'].unique()

# Create a dictionary that maps movie ids to actual ratings for user 1 from the entire dataset
actual_ratings = df.loc[df['userId'] == 1].set_index('movieId')['rating'].to_dict()

# Predict ratings for all movies that user 1 hasn't rated yet
recommendations = []
for movie_id in all_movie_ids:
    if movie_id in rated_movies:
        # Predict rating
        prediction = model.predict(uid=1, iid=movie_id)
        recommendations.append((movie_id, prediction.est))

# Sort the predictions by the estimated rating
recommendations.sort(key=lambda x: x[1], reverse=True)

# Get top N recommendations (e.g., top 10)
top_n_recommendations = recommendations[:50]

# Print the recommended movie ids and their predicted ratings and actual ratings
for movie_id, rating in top_n_recommendations:
    print(f"Movie ID: {movie_id}, Predicted Rating: {rating}, Actual Rating: {actual_ratings.get(movie_id, 0)}")

# Get the lists of actual ratings and predicted ratings from the top N recommendations
actual_ratings_list = [actual_ratings.get(movie_id, 0) for movie_id, rating in top_n_recommendations]
predicted_ratings_list = [rating for movie_id, rating in top_n_recommendations]


plt.scatter(range(1, len(actual_ratings_list) + 1), actual_ratings_list, color='blue', label='Actual Ratings')
plt.plot(range(1, len(predicted_ratings_list) + 1), predicted_ratings_list, color='red', label='Predicted Values')
plt.xlabel('Data Point')
plt.ylabel('Rating')
plt.title('Actual Ratings vs Predicted Values for Item-Based CF')

plt.ylim((0, 5.5))
plt.xlim((0, 50.5))
plt.legend()
plt.show()

print(f"Training Time: {training_time} seconds")
