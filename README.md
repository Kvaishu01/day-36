# 🎬 Day 36 — NMF (Non-Negative Matrix Factorization) Movie Recommender

This project demonstrates **NMF (Non-Negative Matrix Factorization)**, a matrix factorization technique widely used in **recommender systems** and **topic modeling**.

## 📌 How it works
- User-movie ratings are stored in a matrix.
- NMF factorizes this into:
  - **W (user features)**
  - **H (movie features)**
- By reconstructing the matrix, we can predict ratings for unseen movies and recommend the top ones.

## 🚀 Run the app
```bash
streamlit run NMF_Recommender_App.py
