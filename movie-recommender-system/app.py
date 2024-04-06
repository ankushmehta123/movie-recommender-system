import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle

# Load the preprocessed movie data
new_df = pickle.load(open('movies.pkl', 'rb'))

# Vectorization
cv = CountVectorizer(max_features=5000, stop_words='english')
vectors = cv.fit_transform(new_df['tags']).toarray()

# Cosine Similarity
similarity = cosine_similarity(vectors)

def recommend(movie):
    movie_index = new_df[new_df['title'] == movie].index[0]
    distances = similarity[movie_index]
    movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x:x[1])[1:6]

    recommended_movies = [new_df.iloc[i[0]].title for i in movies_list]
    return recommended_movies

# Streamlit UI
st.title("Movie Recommendation System")

selected_movie = st.selectbox("Select a movie:", new_df['title'].values)

if st.button("Recommend"):
    recommended_movies = recommend(selected_movie)
    st.write("Recommended Movies:")
    for movie in recommended_movies:
        st.write(movie)
