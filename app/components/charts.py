# app/components/charts.py
"""
Chart creation components
"""

import plotly.express as px
import plotly.graph_objects as go
import pandas as pd


def create_score_chart(anime_df):
    """Create score distribution chart"""
    fig = px.histogram(
        anime_df,
        x='Score',
        nbins=50,
        title='Anime Score Distribution',
        labels={'Score': 'Score', 'count': 'Number of Anime'},
        color_discrete_sequence=['#667eea']
    )
    
    fig.update_layout(
        showlegend=False,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(size=12),
        title_font=dict(size=16, family='Arial Black')
    )
    
    return fig


def create_genre_chart(anime_df, top_n=15):
    """Create genre frequency chart"""
    genres = anime_df['Genres'].str.split(',').explode().str.strip()
    genre_counts = genres.value_counts().head(top_n)
    
    fig = px.bar(
        x=genre_counts.values,
        y=genre_counts.index,
        orientation='h',
        title=f'Top {top_n} Anime Genres',
        labels={'x': 'Count', 'y': 'Genre'},
        color=genre_counts.values,
        color_continuous_scale='Viridis'
    )
    
    fig.update_layout(
        showlegend=False,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(size=12),
        title_font=dict(size=16, family='Arial Black')
    )
    
    return fig


def create_type_chart(anime_df):
    """Create anime type distribution pie chart"""
    type_counts = anime_df['Type'].value_counts()
    
    fig = px.pie(
        values=type_counts.values,
        names=type_counts.index,
        title='Anime Type Distribution',
        hole=0.4,
        color_discrete_sequence=px.colors.sequential.RdBu
    )
    
    fig.update_layout(
        font=dict(size=12),
        title_font=dict(size=16, family='Arial Black')
    )
    
    return fig


def create_correlation_heatmap(anime_df):
    """Create correlation heatmap"""
    numeric_cols = ['Score', 'Episodes', 'Members', 'Favorites', 'Scored By']
    numeric_cols = [col for col in numeric_cols if col in anime_df.columns]
    
    corr_matrix = anime_df[numeric_cols].corr()
    
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        colorscale='RdBu',
        zmid=0,
        text=corr_matrix.values.round(2),
        texttemplate='%{text}',
        textfont={"size": 10}
    ))
    
    fig.update_layout(
        title='Feature Correlation Heatmap',
        font=dict(size=12),
        title_font=dict(size=16, family='Arial Black'),
        width=600,
        height=500
    )
    
    return fig
# ============================================================================