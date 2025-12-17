# app/components/anime_card.py
"""
Anime card display components
"""

import streamlit as st
import pandas as pd


def display_anime_card(anime_row, show_scores=False, card_key=None):
    """
    Display a single anime card with image and details
    
    Args:
        anime_row: Anime data (pandas Series or dict)
        show_scores: Whether to show similarity/prediction scores
        card_key: Unique key for the card (for Streamlit state)
    """
    # Convert to dict if Series
    if isinstance(anime_row, pd.Series):
        anime = anime_row.to_dict()
    else:
        anime = anime_row
    
    # Create two columns: image and details
    col1, col2 = st.columns([1, 3])
    
    with col1:
        # Display image
        image_url = anime.get('Image URL', '')
        if image_url and pd.notna(image_url):
            st.image(image_url, use_container_width=True)
        else:
            st.image("https://via.placeholder.com/200x300?text=No+Image", 
                    use_container_width=True)
    
    with col2:
        # Title
        st.markdown(f"### 🎬 {anime['Name']}")
        
        # Score badge
        score = anime.get('Score', 0)
        st.markdown(f"**⭐ Score:** {score:.2f}/10")
        
        # Basic info
        anime_type = anime.get('Type', 'Unknown')
        episodes = anime.get('Episodes', 'N/A')
        st.write(f"**📺 Type:** {anime_type} | **Episodes:** {episodes}")
        
        # Genres
        genres = anime.get('Genres', '')
        if genres and pd.notna(genres):
            genre_list = [g.strip() for g in str(genres).split(',')[:5]]
            st.markdown("**🎭 Genres:** " + ", ".join(f"`{g}`" for g in genre_list))
        
        # Additional scores
        if show_scores:
            score_cols = st.columns(3)
            
            if 'similarity_score' in anime:
                with score_cols[0]:
                    st.metric("Similarity", f"{anime['similarity_score']:.3f}")
            
            if 'cb_score' in anime:
                with score_cols[0]:
                    st.metric("CB Score", f"{anime['cb_score']:.3f}")
            
            if 'cf_score' in anime:
                with score_cols[1]:
                    st.metric("CF Score", f"{anime['cf_score']:.3f}")
            
            if 'hybrid_score' in anime:
                with score_cols[2]:
                    st.metric("Hybrid", f"{anime['hybrid_score']:.3f}")
            
            if 'predicted_rating' in anime:
                with score_cols[1]:
                    st.metric("Predicted", f"{anime['predicted_rating']:.2f}")


def display_anime_grid(anime_list, columns=2, show_scores=False):
    """
    Display multiple anime in a grid layout
    
    Args:
        anime_list: List or DataFrame of anime
        columns: Number of columns in grid
        show_scores: Whether to show similarity/prediction scores
    """
    if isinstance(anime_list, pd.DataFrame):
        anime_list = [row for _, row in anime_list.iterrows()]
    
    # Display in grid
    for i in range(0, len(anime_list), columns):
        cols = st.columns(columns)
        
        for j in range(columns):
            idx = i + j
            if idx < len(anime_list):
                with cols[j]:
                    with st.container():
                        display_anime_card(
                            anime_list[idx], 
                            show_scores=show_scores,
                            card_key=f"card_{idx}"
                        )
                        st.markdown("---")


# ============================================================================
