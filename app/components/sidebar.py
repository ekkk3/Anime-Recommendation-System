# app/components/sidebar.py
"""
Sidebar components
"""

import streamlit as st


def create_sidebar(page_options):
    """
    Create sidebar with navigation and stats
    
    Args:
        page_options: List of page names
        
    Returns:
        Selected page
    """
    with st.sidebar:
        # Logo
        st.image("https://img.icons8.com/fluency/96/anime.png", width=80)
        
        # Title
        st.title("🎯 Navigation")
        
        # Page selection
        page = st.radio(
            "Choose a page:",
            page_options,
            label_visibility="collapsed"
        )
        
        return page


def display_stats(anime_df):
    """
    Display dataset statistics in sidebar
    
    Args:
        anime_df: Anime dataframe
    """
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📈 Quick Stats")
    
    col1, col2 = st.sidebar.columns(2)
    
    with col1:
        st.metric("Anime", f"{len(anime_df):,}")
    
    with col2:
        st.metric("Avg Score", f"{anime_df['Score'].mean():.2f}")
    
    # Genres count
    genres_count = len(anime_df['Genres'].str.split(',').explode().unique())
    st.sidebar.metric("Genres", genres_count)
