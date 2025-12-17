"""
Utility functions for anime recommendation system
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
from typing import List, Dict, Tuple, Optional
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def save_visualization(fig, filename: str, output_dir: str = "data/visualizations"):
    """
    Save matplotlib figure
    
    Args:
        fig: Matplotlib figure
        filename: Output filename
        output_dir: Output directory
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    filepath = output_path / filename
    fig.savefig(filepath, dpi=300, bbox_inches='tight')
    logger.info(f"Visualization saved to {filepath}")
    plt.close(fig)


def plot_score_distribution(anime_df: pd.DataFrame, save: bool = True):
    """
    Plot anime score distribution
    
    Args:
        anime_df: Anime dataframe
        save: Whether to save the plot
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    sns.histplot(data=anime_df, x='Score', bins=50, kde=True, ax=ax)
    ax.set_title('Anime Score Distribution', fontsize=16, fontweight='bold')
    ax.set_xlabel('Score', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.grid(alpha=0.3)
    
    if save:
        save_visualization(fig, 'rating_distribution.png')
    
    return fig


def plot_genre_distribution(anime_df: pd.DataFrame, top_n: int = 15, save: bool = True):
    """
    Plot genre frequency distribution
    
    Args:
        anime_df: Anime dataframe
        top_n: Number of top genres to show
        save: Whether to save the plot
    """
    # Extract all genres
    genres = anime_df['Genres'].str.split(',').explode().str.strip()
    genre_counts = genres.value_counts().head(top_n)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    genre_counts.plot(kind='barh', ax=ax, color='#667eea')
    ax.set_title(f'Top {top_n} Anime Genres', fontsize=16, fontweight='bold')
    ax.set_xlabel('Count', fontsize=12)
    ax.set_ylabel('Genre', fontsize=12)
    ax.grid(alpha=0.3, axis='x')
    
    if save:
        save_visualization(fig, 'genre_distribution.png')
    
    return fig


def plot_top_anime(anime_df: pd.DataFrame, top_n: int = 20, 
                  metric: str = 'Score', save: bool = True):
    """
    Plot top anime by given metric
    
    Args:
        anime_df: Anime dataframe
        top_n: Number of top anime to show
        metric: Metric to sort by ('Score', 'Members', 'Favorites')
        save: Whether to save the plot
    """
    top_anime = anime_df.nlargest(top_n, metric)
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    y_pos = np.arange(len(top_anime))
    ax.barh(y_pos, top_anime[metric], color='#764ba2')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_anime['Name'], fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel(metric, fontsize=12)
    ax.set_title(f'Top {top_n} Anime by {metric}', fontsize=16, fontweight='bold')
    ax.grid(alpha=0.3, axis='x')
    
    if save:
        save_visualization(fig, f'top_anime_{metric.lower()}.png')
    
    return fig


def plot_correlation_heatmap(anime_df: pd.DataFrame, save: bool = True):
    """
    Plot correlation heatmap of numeric features
    
    Args:
        anime_df: Anime dataframe
        save: Whether to save the plot
    """
    # Select numeric columns
    numeric_cols = ['Score', 'Episodes', 'Members', 'Favorites', 'Scored By']
    numeric_cols = [col for col in numeric_cols if col in anime_df.columns]
    
    correlation_matrix = anime_df[numeric_cols].corr()
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm',
                center=0, square=True, linewidths=1, ax=ax)
    ax.set_title('Feature Correlation Heatmap', fontsize=16, fontweight='bold')
    
    if save:
        save_visualization(fig, 'correlation_heatmap.png')
    
    return fig


def plot_type_distribution(anime_df: pd.DataFrame, save: bool = True):
    """
    Plot anime type distribution (pie chart)
    
    Args:
        anime_df: Anime dataframe
        save: Whether to save the plot
    """
    type_counts = anime_df['Type'].value_counts()
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    colors = plt.cm.Set3(np.linspace(0, 1, len(type_counts)))
    ax.pie(type_counts, labels=type_counts.index, autopct='%1.1f%%',
           startangle=90, colors=colors)
    ax.set_title('Anime Type Distribution', fontsize=16, fontweight='bold')
    
    if save:
        save_visualization(fig, 'type_distribution.png')
    
    return fig


def create_all_visualizations(anime_df: pd.DataFrame, output_dir: str = "data/visualizations"):
    """
    Create all standard visualizations
    
    Args:
        anime_df: Anime dataframe
        output_dir: Output directory for plots
    """
    logger.info("Creating all visualizations...")
    
    plot_score_distribution(anime_df, save=True)
    plot_genre_distribution(anime_df, save=True)
    plot_top_anime(anime_df, metric='Score', save=True)
    plot_top_anime(anime_df, metric='Members', save=True)
    plot_correlation_heatmap(anime_df, save=True)
    plot_type_distribution(anime_df, save=True)
    
    logger.info(f"All visualizations saved to {output_dir}")


def format_anime_info(anime_row: pd.Series) -> str:
    """
    Format anime information as readable string
    
    Args:
        anime_row: Anime data row
        
    Returns:
        Formatted string
    """
    info = f"""
    📺 {anime_row['Name']}
    ⭐ Score: {anime_row['Score']:.2f}/10
    🎭 Genres: {anime_row['Genres']}
    📊 Type: {anime_row['Type']}
    📺 Episodes: {anime_row.get('Episodes', 'N/A')}
    👥 Members: {anime_row.get('Members', 0):,.0f}
    """
    return info.strip()


def calculate_similarity_threshold(similarity_matrix: np.ndarray, 
                                  percentile: float = 75) -> float:
    """
    Calculate similarity threshold based on percentile
    
    Args:
        similarity_matrix: Similarity matrix
        percentile: Percentile for threshold
        
    Returns:
        Threshold value
    """
    # Get upper triangle (excluding diagonal)
    upper_triangle = similarity_matrix[np.triu_indices_from(similarity_matrix, k=1)]
    threshold = np.percentile(upper_triangle, percentile)
    
    logger.info(f"Similarity threshold ({percentile}th percentile): {threshold:.3f}")
    return threshold


def filter_by_genres(anime_df: pd.DataFrame, genres: List[str], 
                    match_all: bool = False) -> pd.DataFrame:
    """
    Filter anime by genres
    
    Args:
        anime_df: Anime dataframe
        genres: List of genres to filter
        match_all: If True, anime must have all genres; if False, any genre
        
    Returns:
        Filtered dataframe
    """
    if match_all:
        mask = anime_df['Genres'].apply(
            lambda x: all(genre in str(x) for genre in genres)
        )
    else:
        mask = anime_df['Genres'].apply(
            lambda x: any(genre in str(x) for genre in genres)
        )
    
    filtered = anime_df[mask]
    logger.info(f"Filtered {len(filtered)} anime with genres: {genres}")
    return filtered


def get_popular_anime(anime_df: pd.DataFrame, min_members: int = 100000, 
                     top_n: int = 50) -> pd.DataFrame:
    """
    Get popular anime based on member count
    
    Args:
        anime_df: Anime dataframe
        min_members: Minimum member count
        top_n: Number of top anime to return
        
    Returns:
        Popular anime dataframe
    """
    popular = anime_df[anime_df['Members'] >= min_members].nlargest(top_n, 'Members')
    logger.info(f"Found {len(popular)} popular anime")
    return popular


def export_recommendations_to_csv(recommendations: pd.DataFrame, 
                                 filename: str, output_dir: str = "data/processed"):
    """
    Export recommendations to CSV
    
    Args:
        recommendations: Recommendations dataframe
        filename: Output filename
        output_dir: Output directory
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    filepath = output_path / filename
    recommendations.to_csv(filepath, index=False)
    logger.info(f"Recommendations exported to {filepath}")


def load_config(config_path: str = "config.json") -> Dict:
    """
    Load configuration from JSON file
    
    Args:
        config_path: Path to config file
        
    Returns:
        Configuration dictionary
    """
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        logger.info(f"Configuration loaded from {config_path}")
        return config
    except FileNotFoundError:
        logger.warning(f"Config file not found: {config_path}")
        return {}


def save_config(config: Dict, config_path: str = "config.json"):
    """
    Save configuration to JSON file
    
    Args:
        config: Configuration dictionary
        config_path: Path to save config
    """
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)
    logger.info(f"Configuration saved to {config_path}")


def print_dataset_summary(anime_df: pd.DataFrame, ratings_df: Optional[pd.DataFrame] = None):
    """
    Print comprehensive dataset summary
    
    Args:
        anime_df: Anime dataframe
        ratings_df: Optional ratings dataframe
    """
    print("\n" + "=" * 70)
    print("DATASET SUMMARY")
    print("=" * 70)
    
    print(f"\n📺 ANIME DATA:")
    print(f"  Total anime: {len(anime_df):,}")
    print(f"  Average score: {anime_df['Score'].mean():.2f}")
    print(f"  Score range: {anime_df['Score'].min():.1f} - {anime_df['Score'].max():.1f}")
    print(f"  Total genres: {len(anime_df['Genres'].str.split(',').explode().unique())}")
    print(f"  Types: {', '.join(anime_df['Type'].unique())}")
    
    if ratings_df is not None:
        print(f"\n⭐ RATINGS DATA:")
        print(f"  Total ratings: {len(ratings_df):,}")
        print(f"  Unique users: {ratings_df['user_id'].nunique():,}")
        print(f"  Unique anime: {ratings_df['anime_id'].nunique():,}")
        print(f"  Average rating: {ratings_df['rating'].mean():.2f}")
        print(f"  Rating range: {ratings_df['rating'].min():.0f} - {ratings_df['rating'].max():.0f}")
        
        sparsity = 1 - (len(ratings_df) / (ratings_df['user_id'].nunique() * 
                                           ratings_df['anime_id'].nunique()))
        print(f"  Sparsity: {sparsity * 100:.2f}%")
    
    print("\n" + "=" * 70 + "\n")


if __name__ == "__main__":
    from config import ANIME_CLEANED
    
    # Load data
    anime_df = pd.read_csv(ANIME_CLEANED)
    
    # Print summary
    print_dataset_summary(anime_df)
    
    # Create visualizations
    create_all_visualizations(anime_df)