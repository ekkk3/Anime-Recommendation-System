import streamlit as st
from utils import display_anime_card

anime_df = st.session_state.anime_df

st.markdown("### 🔍 Search Anime")
search_query = st.text_input("", placeholder="Enter anime name...")

if search_query:
    results = anime_df[anime_df['Name'].str.contains(search_query, case=False, na=False)].sort_values('Score', ascending=False)
    st.markdown(f"### 📋 Found {len(results)} results")
    if len(results) > 0:
        for _, anime in results.iterrows():
            with st.container():
                st.markdown('<div class="anime-card">', unsafe_allow_html=True)
                display_anime_card(anime)
                st.markdown('</div>', unsafe_allow_html=True)
                st.markdown("---")
    else:
        st.info("😔 No anime found")