# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 18:18:37 2020

@author: shris
"""
# continuation of previous code

is_model = Recommenders.item_similarity_recommender_py()
is_model.create(train_data, 'user_id', 'song')

user_id = users[4]
user_items = is_model.get_user_items(user_id)
#now training for specific user
print("------------------------------------------------------------------------------------")
print("Training data songs for the user userid: %s:" % user_id)
print("------------------------------------------------------------------------------------")

for user_item in user_items:
    print(user_item)
    
print("----------------------------------------------------------------------")
print("Recommendation process going on:")
print("----------------------------------------------------------------------")

#Recommend songs for the user using personalized model
is_model.recommend(user_id)
# now we will get different recommendations for different user.
user_id = users[8]
#Fill in the code here
user_items = is_model.get_user_items(user_id)
#
print("------------------------------------------------------------------------------------")
print("Training data songs for the user userid: %s:" % user_id)
print("------------------------------------------------------------------------------------")

for user_item in user_items:
    print(user_item)

print("----------------------------------------------------------------------")
print("Recommendation process going on:")
print("----------------------------------------------------------------------")

#Recommend songs for the user using personalized model
is_model.recommend(user_id)

# to get similar songs
is_model.get_similar_items(['Mockingbird - Eminem'])

song = 'Clocks - Coldplay'
###Fill in the code here
is_model.get_similar_items([song])