import streamlit as st
import plotly.express as px

anime_df = st.session_state.anime_df

st.markdown("### 📊 Analytics")

tab1, tab2, tab3 = st.tabs(["📈 Scores", "🎭 Genres", "📺 Types"])

with tab1:
    fig = px.histogram(anime_df, x='Score', nbins=50, color_discrete_sequence=['#667eea'])
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    genres = anime_df['Genres'].str.split(',').explode().str.strip()
    genre_counts = genres.value_counts().head(15)
    fig = px.bar(x=genre_counts.values, y=genre_counts.index, orientation='h', color_discrete_sequence=['#764ba2'])
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    type_counts = anime_df['Type'].value_counts()
    fig = px.pie(values=type_counts.values, names=type_counts.index, hole=0.4)
    st.plotly_chart(fig, use_container_width=True)