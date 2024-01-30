import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time
from sklearn.metrics import mean_squared_error
from scipy import sparse
from surprise import Reader, Dataset, SVD, accuracy
from surprise.model_selection import train_test_split
from contextlib import redirect_stdout

print("Loading the data...")

if not os.path.isfile("ratings_1.csv"):
    ratings_1 = open("ratings_1.csv", mode="w")
    ratings_2 = open("ratings_2.csv", mode="w")
    ratings_3 = open("ratings_3.csv", mode="w")
    ratings_4 = open("ratings_4.csv", mode="w")
    

    ratings_files = ["combined_data_1.txt", "combined_data_2.txt", "combined_data_3.txt", "combined_data_4.txt"]
    ratings_csvs = [ratings_1, ratings_2, ratings_3, ratings_4]

    for file, ratings in zip(ratings_files,ratings_csvs):
        movie_id = 0
        with open(file) as f:
            for line in f:
                line = line.strip()
                line = line.split(",")


                if line[0].endswith(":"):
                    movie_id = line[0].replace(":","")
                else:
                    line.insert(0,str(movie_id))
                    ratings.write(str(",".join(line)))
                    ratings.write("\n")

    ratings_1.close()
    ratings_2.close()
    ratings_3.close()
    ratings_4.close()


df_1 = pd.read_csv("ratings_1.csv",header=None,sep=",",names=["Movie_ID","User_ID","Rating","Date"]) 
df_2 = pd.read_csv("ratings_2.csv",header=None,sep=",",names=["Movie_ID","User_ID","Rating","Date"]) 
df_3 = pd.read_csv("ratings_3.csv",header=None,sep=",",names=["Movie_ID","User_ID","Rating","Date"]) 
df_4 = pd.read_csv("ratings_4.csv",header=None,sep=",",names=["Movie_ID","User_ID","Rating","Date"]) 

df = pd.concat([df_1,df_2,df_3,df_4], axis=0, ignore_index=True)

# keep ~1000 users
df = df[df["User_ID"] <= 6000]

#print("Total number of users:", len(np.unique(df["User_ID"])))
#print("Total number of movies:", len(np.unique(df["Movie_ID"])))

print("Training the model and generating the predictions...")

updated_df = df.drop(columns=['Date'], axis=1)

reader = Reader(rating_scale=(1,5))
data = Dataset.load_from_df(updated_df[['User_ID', 'Movie_ID', 'Rating']], reader)

# Build the full trainset (using all the data)
full_trainset = data.build_full_trainset()

# Create and fit the SVD model using the full trainset
svd_model_1 = SVD(n_factors=25, n_epochs=20, lr_all=0.005, reg_all=0.02)
svd_model_1.fit(full_trainset)

# Get predictions for all user-item pairs in the dataset
all_predictions = svd_model_1.test(full_trainset.build_anti_testset())

movie_titles_df = pd.read_csv('movie_titles.csv', usecols=[0, 1, 2], header=None, names=['Movie_ID', 'Year', 'Title'], encoding='ISO-8859-1')

print("Done!")

#
# Finished with the data preperation and the predictions, now we can take user input and give him the recommendation
#

while True:
    # Get user input for the desired user (not the user ID since there are several IDs that are missing, but rather the user with the X lowest ID)
    user_choice = int(input("Enter the desired user (integer) or '-1' to Exit: "))

    if user_choice == -1:
        break
    

    # Find the user with the chosen user with the X lowest ID
    sorted_user_ids = updated_df['User_ID'].value_counts().index.sort_values()
    user_with_chosen_id = sorted_user_ids[user_choice - 1]

    '''
    # Get movies that the user has already rated (so we can exclude them later)
    #movies_rated_by_user = df[df['User_ID'] == user_with_chosen_id]['Movie_ID'].values

    # Create a test set for the specific user
    #test_set_for_user = [(user_with_chosen_id, movie_id, 4.0) for movie_id in np.unique(df['Movie_ID']) if movie_id not in movies_rated_by_user]

    # Get predictions only for the specific user
    #user_chosen_predictions = svd_model_1.test(test_set_for_user)
    '''

    # Get predictions for the user
    user_chosen_predictions = [pred for pred in all_predictions if pred.uid == user_with_chosen_id]

    # Get the top 10 recommendations for the user
    user_chosen_top_10_recommendations = sorted(user_chosen_predictions, key=lambda x: x.est, reverse=True)[:10]

    # Print top 10 recommendations for the user
    print(f"\nTop 10 Recommendations for User {user_choice} (ID = {user_with_chosen_id}):")
    for pred in user_chosen_top_10_recommendations:

        # Look up movie title based on movie ID
        movie_title = movie_titles_df.loc[movie_titles_df['Movie_ID'] == pred.iid, 'Title'].values[0]

        print(f"- {movie_title} (ID: {pred.iid}, Predicted Rating: {pred.est})")
    
    print("")


