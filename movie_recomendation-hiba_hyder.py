#!/usr/bin/env python
# coding: utf-8

# In[1]:


#importing necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import pairwise_distances


# In[2]:


#loading the dataset
movies = pd.read_csv(r"D:\Users\hibah\Desktop\movies.csv")
ratings = pd.read_csv(r"D:\Users\hibah\Desktop\ratings.csv")


# In[3]:


movies.info()


# In[4]:


ratings.info()


# In[5]:


movies.head()


# In[6]:


ratings.head()


# In[7]:


# Merge movie and rating data
movie_ratings = pd.merge(ratings, movies, on='movieId')


# In[8]:


movie_ratings.info()


# In[9]:


movie_ratings=movie_ratings.head(1000000)


# In[10]:


# Create a user-movie rating matrix
user_movie_ratings = movie_ratings.pivot_table(index='userId', columns='title', values='rating')


# In[11]:


# Fill missing values with 0
user_movie_ratings = user_movie_ratings.fillna(0)


# In[12]:


# Transpose the matrix to get movie-user ratings
movie_user_ratings = user_movie_ratings.T


# In[13]:


# Calculate similarity between movies using cosine similarity
movie_similarity = cosine_similarity(movie_user_ratings)


# In[14]:


# Create a DataFrame from the similarity matrix
movie_similarity_df = pd.DataFrame(movie_similarity, index=user_movie_ratings.columns, columns=user_movie_ratings.columns)

# Function to recommend movies for a given movie title
def recommend_movies(movie_title, num_recommendations=5):
    similar_scores = movie_similarity_df[movie_title]
    similar_movies = list(similar_scores.index)
    similar_movies.remove(movie_title)
    top_similar_movies = similar_scores.sort_values(ascending=False).head(num_recommendations)
    return top_similar_movies


# In[15]:


recommend_movies("Toy Story (1995)")


# In[ ]:





# In[ ]:




