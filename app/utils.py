import streamlit as st
import pandas as pd

def display_anime_card(anime_row, show_scores=False):
    col1, col2 = st.columns([1, 3])
    
    with col1:
        if 'Image URL' in anime_row and pd.notna(anime_row['Image URL']):
            st.image(anime_row['Image URL'], use_container_width=True)
        else:
            st.image("https://picsum.photos/200/300", use_container_width=True)
    
    with col2:
        st.markdown(f'<div class="anime-title">🎬 {anime_row["Name"]}</div>', unsafe_allow_html=True)
        st.markdown(f'<span class="anime-score">⭐ {anime_row["Score"]:.2f}/10</span>', unsafe_allow_html=True)
        
        if 'rating' in anime_row and pd.notna(anime_row['rating']):
            st.markdown(f'<span class="anime-score">Your Rating: {anime_row["rating"]}/10</span>', unsafe_allow_html=True)
        
        st.write(f"**Type:** {anime_row['Type']} | **Episodes:** {anime_row.get('Episodes', 'N/A')}")
        
        if 'Genres' in anime_row and pd.notna(anime_row['Genres']):
            genres = anime_row['Genres'].split(',')
            genre_html = ''.join([f'<span class="genre-badge">{g.strip()}</span>' for g in genres[:5]])
            st.markdown(f"**Genres:** {genre_html}", unsafe_allow_html=True)
        
        if show_scores:
            cols = st.columns(4)
            if 'similarity_score' in anime_row:
                cols[0].metric("Similarity", f"{anime_row['similarity_score']:.3f}")
            if 'cb_score' in anime_row:
                cols[1].metric("CB Score", f"{anime_row['cb_score']:.3f}")
            if 'cf_score' in anime_row:
                cols[2].metric("CF Score", f"{anime_row['cf_score']:.3f}")
            if 'hybrid_score' in anime_row:
                cols[3].metric("Hybrid Score", f"{anime_row['hybrid_score']:.3f}")

def display_anime_grid(df, columns=4, show_scores=False):
    num_cols = columns
    for i in range(0, len(df), num_cols):
        cols = st.columns(num_cols)
        for j, idx in enumerate(range(i, min(i + num_cols, len(df)))):
            with cols[j]:
                with st.container():
                    st.markdown('<div class="anime-card">', unsafe_allow_html=True)
                    display_anime_card(df.iloc[idx], show_scores=show_scores)
                    st.markdown('</div>', unsafe_allow_html=True)

def get_recommendations(models, anime_id, model_type, top_n, min_score):
    model = models.get(model_type)
    if model is None:
        st.error(f"❌ {model_type} model not available")
        return pd.DataFrame()
    
    try:
        if model_type == 'hybrid':
            return model.recommend(anime_id=anime_id, top_n=top_n, min_score=min_score)
        else:
            return model.recommend(anime_id, top_n, min_score)
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        return pd.DataFrame()