# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 18:02:28 2020

@author: shris
"""
#run in jupyter notebook
#importing libraries
import pandas
from sklearn.model_selection import train_test_split
import numpy as np
import time
from sklearn.externals import joblib
import Recommenders as Recommenders

## million song dataset has two parts one is triplets other is metadata file
triplets_file = 'https://static.turi.com/datasets/millionsong/10000.txt'
songs_metadata_file = 'https://static.turi.com/datasets/millionsong/song_data.csv'
# the triplet file conatins user id , song id and listen count
#the meta file conatins details about the song i.e  artist , release ,year
song_df_1 = pandas.read_table(triplets_file,header=None)
song_df_1.columns = ['user_id', 'song_id', 'listen_count']
song_df_2 =  pandas.read_csv(songs_metadata_file) 
#merging both and removing duplicates using song ID which is unique
song_df = pandas.merge(song_df_1, song_df_2.drop_duplicates(['song_id']), on="song_id", how="left")

song_df.head()

len(song_df)
song_df = song_df.head(10000)
#combining song with its artist name 
song_df['song'] = song_df['title'].map(str) + " - " + song_df['artist_name']
# calculating the most popular song by calculating the listen count form all users in the dataset and also calculating its percentage with respect to other unique songs.
song_grouped = song_df.groupby(['song']).agg({'listen_count': 'count'}).reset_index()
grouped_sum = song_grouped['listen_count'].sum()
song_grouped['percentage']  = song_grouped['listen_count'].div(grouped_sum)*100
song_grouped.sort_values(['listen_count', 'song'], ascending = [0,1])
users = song_df['user_id'].unique()

len(users)
songs = song_df['song'].unique()
len(songs)
 ## spliting  the dataset into test train 
train_data, test_data = train_test_split(song_df, test_size = 0.20, random_state=0)
print(train_data.head(5))

pm = Recommenders.popularity_recommender_py()
pm.create(train_data, 'user_id', 'song')
## getting the recommended songs for user, here the answer for eevry user will be same as it not a personalised model and is only based on popularity of the song
user_id = users[5]
pm.recommend(user_id)

user_id = users[7]
pm.recommend(user_id)
# can try for as many unique users.