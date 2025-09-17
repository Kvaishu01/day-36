import streamlit as st
import pandas as pd
import numpy as np
from sklearn.decomposition import NMF
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Day 36 - NMF Recommender", layout="wide")
st.title("🎬 Day 36 — Movie Recommender with NMF")

# ---------------------------
# Load dataset
# ---------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("movies.csv")
    return df

movies = load_data()

st.subheader("📂 Sample Movies Dataset")
st.dataframe(movies.head())

# ---------------------------
# Create user-movie matrix
# ---------------------------
ratings = movies.pivot(index="userId", columns="title", values="rating").fillna(0)

# ---------------------------
# NMF model
# ---------------------------
nmf = NMF(n_components=5, random_state=42)
W = nmf.fit_transform(ratings)   # user features
H = nmf.components_              # item features

# ---------------------------
# Recommendation function
# ---------------------------
def recommend_movies(user_id, top_n=5):
    user_index = user_id - 1  # since userId starts at 1
    user_features = W[user_index].reshape(1, -1)
    scores = np.dot(user_features, H)
    scores = scores.flatten()

    # sort by score
    movie_scores = pd.Series(scores, index=ratings.columns)
    recommended = movie_scores.sort_values(ascending=False).head(top_n)
    return recommended

# ---------------------------
# User input
# ---------------------------
user_choice = st.selectbox("Select a user (ID)", movies["userId"].unique())
top_n = st.slider("Number of recommendations", 1, 10, 5)

if st.button("Recommend Movies"):
    recs = recommend_movies(user_choice, top_n)
    st.subheader(f"🎥 Recommended Movies for User {user_choice}")
    st.table(recs.reset_index().rename(columns={"index": "Movie", 0: "Score"}))

st.success("✅ NMF-based recommendation complete!")
