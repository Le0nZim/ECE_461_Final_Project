# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 01:06:32 2024

@author: DaNi
"""

"""

"""
#%%
import pandas as pd

# Import the datasets
tracks = pd.read_csv("tracks.csv")
cust = pd.read_csv("cust.csv")
music = pd.read_csv("music.csv")

# Make the correct data format
# Merge datasets
merged_data1 = pd.merge(tracks, music, on='TrackId', how='left')
merged_data = pd.merge(merged_data1, cust, on='CustID', how='left')

"""
    First Merge (pd.merge(tracks, music, on='TrackId', how='left')):
    The pd.merge function is used to merge the 'tracks' and 'music' DataFrames based on the common column 'TrackId'.
    The on='TrackId' parameter specifies that the merging should be based on the 'TrackId' column.
    The how='left' parameter indicates a left join, meaning all records from the 'tracks' DataFrame will be included 
    in the result, and matching records from the 'music' DataFrame will be added where available.

    Second Merge (pd.merge(merged_data, cust, on='CustID', how='left')):
    The result from the first merge is then further merged with the 'cust' DataFrame based on the common column 'CustID'.
    The on='CustID' parameter specifies that the merging should be based on the 'CustID' column.
    Again, the how='left' parameter indicates a left join, preserving all records from the 'merged_data' DataFrame 
    and adding matching records from the 'cust' DataFrame where available.
"""


# Group by customer id and artist id, and calculate playscount
playscount_data = merged_data.groupby(['CustID', 'TrackId']).size().reset_index(name='playscount')

"""
    Grouping and Calculating Playscount:
    The groupby method is used to group the merged_data DataFrame by the columns 'CustID' and 'Artist'.
    The size() method counts the number of occurrences in each group, effectively calculating the playscount 
    for each combination of 'CustID' and 'Artist'.
    The reset_index(name='playscount') part resets the index of the resulting DataFrame and names the count 
    column as 'playscount'.
"""
# Select relevant columns from cust dataset
cust_new = cust[['CustID', 'Gender', 'zip', 'SignDate', 'Level']]

"""
    Selecting Relevant Columns from 'cust' Dataset:
    The code selects specific columns ('CustID', 'Gender', 'Address', 'zip', 'SignDate') from the 'cust' 
    DataFrame and creates a new DataFrame called cust_new.
"""
# Merge playscount data with result data
cust_new = pd.merge(cust_new, playscount_data, on='CustID', how='left')

"""
    Merging Playscount Data with 'cust_new':
    The cust_new DataFrame is then merged with the playscount data based on the 'CustID' column using a left join.
"""
# Fill NaN values with 0 in playscount column
cust_new['playscount'] = cust_new['playscount'].fillna(0).astype(int)

"""
    Filling NaN Values in 'playscount' Column:
    NaN values in the 'playscount' column (resulting from the left join) are filled with 0 using the fillna(0) method.
    The column is then cast to integer type using astype(int).
"""

# Make a csv file to pass to spark
csv_file_path = 'subset1.csv'
cust_new.to_csv(csv_file_path, index=False)
#%%