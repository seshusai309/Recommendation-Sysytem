import streamlit as st
import pickle 
import numpy as np
import pandas as pd

# Configure page
st.set_page_config(
    page_title="BookVoyage | AI-Powered Recommender",
    page_icon="📚",
    layout="wide"
)

# Custom CSS for professional look
st.markdown("""
<style>
    :root {
        --primary: #2c3e50;
        --secondary: #3498db;
        --accent: #e74c3c;
        --light: #ecf0f1;
        --dark: #1a1a1a;
    }
    
    html, body, [class*="css"] {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    .header {
        background: linear-gradient(135deg, var(--primary), var(--dark));
        color: white;
        padding: 2rem;
        border-radius: 0 0 20px 20px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 20px rgba(0,0,0,0.1);
    }
    
    .header-title {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    
    .header-subtitle {
        font-size: 1.1rem;
        opacity: 0.9;
    }
    
    .book-card {
        background: white;
        border-radius: 10px;
        padding: 1rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        transition: transform 0.3s ease;
        height: 100%;
        display: flex;
        flex-direction: column;
    }
    
    .book-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.15);
    }
    
    .book-cover {
        width: 100%;
        height: 200px;
        object-fit: contain;
        border-radius: 8px;
        background: #f5f5f5;
        margin-bottom: 1rem;
    }
    
    .book-title {
        font-weight: 600;
        margin: 0.5rem 0;
        color: var(--dark);
        font-size: 1rem;
        line-height: 1.3;
    }
    
    .book-author {
        color: #666;
        font-size: 0.9rem;
        margin-bottom: 0.5rem;
    }
    
    .sidebar .sidebar-content {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
    }
    
    .stButton>button {
        background: var(--secondary);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: 600;
        transition: all 0.3s;
    }
    
    .stButton>button:hover {
        background: var(--primary);
        transform: translateY(-2px);
    }
    
    .section-title {
        color: var(--primary);
        font-size: 1.5rem;
        margin: 1.5rem 0 1rem;
        border-bottom: 2px solid var(--secondary);
        padding-bottom: 0.5rem;
    }
    
    .badge {
        background: var(--accent);
        color: white;
        padding: 0.2rem 0.5rem;
        border-radius: 50px;
        font-size: 0.8rem;
        font-weight: 600;
        display: inline-block;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Header section
st.markdown("""
<div class="header">
    <div class="header-title">BookVoyage</div>
    <div class="header-subtitle">AI-Powered Book Recommendation Engine</div>
    <div class="badge">Portfolio Project</div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<p style="font-size: 1.1rem; color: #555;">
    This intelligent recommender system uses collaborative filtering to suggest books from our extensive catalog. 
    Discover your next favorite read with our personalized recommendations or browse our top-rated collection.
</p>
""", unsafe_allow_html=True)

# Load models
@st.cache_data
def load_data():
    popular = pickle.load(open('popular.pkl', 'rb'))
    books = pickle.load(open('books.pkl', 'rb'))
    pt = pickle.load(open('pt.pkl', 'rb'))
    similarity_scores = pickle.load(open('similarity_scores.pkl', 'rb'))
    return popular, books, pt, similarity_scores

try:
    popular, books, pt, similarity_scores = load_data()
except Exception as e:
    st.error(f"Error loading data: {str(e)}")
    st.stop()

# Check if required columns exist in popular DataFrame
required_columns = ['Book-Title', 'Book-Author', 'Image-URL-M', 'avg_rating']
for col in required_columns:
    if col not in popular.columns:
        st.error(f"Missing required column in data: {col}")
        st.stop()

# Sidebar controls
st.sidebar.markdown("## 🔍 Navigation")

# Top 50 Books section
if st.sidebar.checkbox("Show Top 50 Books", True):
    st.markdown('<div class="section-title">📊 Top Rated Books</div>', unsafe_allow_html=True)
    
    cols_per_row = 5
    num_rows = 10
    
    for row in range(num_rows):
        cols = st.columns(cols_per_row)
        for col in range(cols_per_row):
            idx = row * cols_per_row + col
            if idx < len(popular):
                book = popular.iloc[idx]
                with cols[col]:
                    # Safely get rating info - use 0 if not available
                    rating_info = ""
                    if 'avg_rating' in popular.columns:
                        rating = book.get('avg_rating', 0)
                        rating_info = f"⭐ {float(rating):.1f}"
                    
                    # Check if num_ratings exists
                    if 'num_ratings' in popular.columns:
                        num_ratings = book.get('num_ratings', 0)
                        rating_info += f" • {int(num_ratings)} reviews"
                    
                    st.markdown(f"""
                    <div class="book-card">
                        <img src="{book['Image-URL-M']}" class="book-cover" onerror="this.src='https://via.placeholder.com/150x200?text=No+Cover'">
                        <div class="book-title">{book['Book-Title']}</div>
                        <div class="book-author">by {book['Book-Author']}</div>
                        <div style="margin-top: auto; font-size: 0.8rem;">
                            {rating_info}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

# Recommendation function
def recommend(book_name):
    try:
        index = np.where(pt.index == book_name)[0][0]
        similar_items = sorted(list(enumerate(similarity_scores[index])), key=lambda x: x[1], reverse=True)[1:11]
        data = []
        for i in similar_items:
            temp_df = books[books['Book-Title'] == pt.index[i[0]]]
            book_info = [
                temp_df.drop_duplicates('Book-Title')['Book-Title'].values[0],
                temp_df.drop_duplicates('Book-Title')['Book-Author'].values[0],
                temp_df.drop_duplicates('Book-Title')['Image-URL-M'].values[0]
            ]
            # Add rating if available
            if 'avg_rating' in temp_df.columns:
                book_info.append(temp_df.drop_duplicates('Book-Title')['avg_rating'].values[0])
            else:
                book_info.append(0)
            data.append(book_info)
        return data
    except Exception as e:
        st.error(f"Error generating recommendations: {str(e)}")
        return []

# Recommendation section
st.sidebar.markdown("## 🔮 Get Recommendations")
book_list = pt.index.values
selected_book = st.sidebar.selectbox("Select a book you enjoy", book_list)

if st.sidebar.button("Find Similar Books"):
    recommendations = recommend(selected_book)
    
    if not recommendations:
        st.warning("No recommendations found for the selected book.")
    else:
        st.markdown(f'<div class="section-title">✨ Recommended based on "{selected_book}"</div>', unsafe_allow_html=True)
        
        cols = st.columns(5)
        for idx in range(5):
            if idx < len(recommendations):
                book = recommendations[idx]
                with cols[idx]:
                    st.markdown(f"""
                    <div class="book-card">
                        <img src="{book[2]}" class="book-cover" onerror="this.src='https://via.placeholder.com/150x200?text=No+Cover'">
                        <div class="book-title">{book[0]}</div>
                        <div class="book-author">by {book[1]}</div>
                        <div style="margin-top: auto; font-size: 0.8rem;">
                            ⭐ {float(book[3]):.1f} average rating
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

# Footer
st.markdown("""
<hr style="margin: 2rem 0; border: 0.5px solid #eee;">
<div style="text-align: center; color: #777; font-size: 0.9rem;">
    <p>BookVoyage Recommender System • Built By Seshu</p>
    <p>Built with Streamlit and collaborative filtering</p>
    <a href="https://github.com/seshusai309/Recommendation-Sysytem" target="_blank" style="display: inline-block; margin-top: 10px; padding: 8px 18px; background-color: #24292f; color: #fff; border-radius: 5px; text-decoration: none; font-weight: 500; font-size: 0.95rem;">
        ⭐ View on GitHub
    </a>
</div>
""", unsafe_allow_html=True)