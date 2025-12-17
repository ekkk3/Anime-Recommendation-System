"""
Data loading and sampling utilities
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Tuple, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_anime_data(file_path: str, cleaned: bool = False) -> pd.DataFrame:
    """
    Load anime dataset
    
    Args:
        file_path: Path to anime CSV file
        cleaned: If True, expects cleaned data format
        
    Returns:
        DataFrame with anime data
    """
    logger.info(f"Loading anime data from {file_path}")
    
    try:
        df = pd.read_csv(file_path)
        logger.info(f"Loaded {len(df)} anime entries")
        
        if cleaned:
            logger.info("Data is already cleaned")
        else:
            logger.info("Raw data loaded - consider cleaning")
        
        return df
    
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        raise
    except Exception as e:
        logger.error(f"Error loading anime data: {str(e)}")
        raise


def load_ratings_data(file_path: str, nrows: Optional[int] = None) -> pd.DataFrame:
    """
    Load ratings dataset
    
    Args:
        file_path: Path to ratings CSV file
        nrows: Number of rows to load (for testing with large files)
        
    Returns:
        DataFrame with ratings data
    """
    logger.info(f"Loading ratings data from {file_path}")
    
    try:
        if nrows:
            df = pd.read_csv(file_path, nrows=nrows)
            logger.info(f"Loaded {len(df)} ratings (limited to {nrows} rows)")
        else:
            df = pd.read_csv(file_path)
            logger.info(f"Loaded {len(df)} ratings")
        
        return df
    
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        raise
    except Exception as e:
        logger.error(f"Error loading ratings data: {str(e)}")
        raise


def sample_ratings(
    ratings_df: pd.DataFrame, 
    sample_size: int = 5_000_000,
    min_user_ratings: int = 5,
    min_anime_ratings: int = 10,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Sample ratings data intelligently
    
    Args:
        ratings_df: Full ratings dataframe
        sample_size: Target number of ratings
        min_user_ratings: Minimum ratings per user
        min_anime_ratings: Minimum ratings per anime
        random_state: Random seed
        
    Returns:
        Sampled ratings dataframe
    """
    logger.info(f"Sampling {sample_size:,} ratings from {len(ratings_df):,} total")
    
    # Remove watching status (rating = 0)
    ratings_df = ratings_df[ratings_df['rating'] > 0].copy()
    logger.info(f"After removing watching status: {len(ratings_df):,} ratings")
    
    # Filter users with minimum ratings
    user_counts = ratings_df['user_id'].value_counts()
    valid_users = user_counts[user_counts >= min_user_ratings].index
    ratings_df = ratings_df[ratings_df['user_id'].isin(valid_users)]
    logger.info(f"After user filtering: {len(ratings_df):,} ratings from {len(valid_users):,} users")
    
    # Filter anime with minimum ratings
    anime_counts = ratings_df['anime_id'].value_counts()
    valid_anime = anime_counts[anime_counts >= min_anime_ratings].index
    ratings_df = ratings_df[ratings_df['anime_id'].isin(valid_anime)]
    logger.info(f"After anime filtering: {len(ratings_df):,} ratings for {len(valid_anime):,} anime")
    
    # Sample if still too large
    if len(ratings_df) > sample_size:
        ratings_df = ratings_df.sample(n=sample_size, random_state=random_state)
        logger.info(f"Sampled down to {len(ratings_df):,} ratings")
    
    return ratings_df.reset_index(drop=True)


def load_animelist_data(file_path: str) -> pd.DataFrame:
    """
    Load animelist (user watch history) dataset
    
    Args:
        file_path: Path to animelist CSV file
        
    Returns:
        DataFrame with user watch history
    """
    logger.info(f"Loading animelist data from {file_path}")
    
    try:
        df = pd.read_csv(file_path)
        logger.info(f"Loaded {len(df)} watch history entries")
        return df
    
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        raise
    except Exception as e:
        logger.error(f"Error loading animelist data: {str(e)}")
        raise


def create_user_item_matrix(
    ratings_df: pd.DataFrame,
    save_path: Optional[str] = None
) -> Tuple[np.ndarray, dict, dict]:
    """
    Create user-item rating matrix
    
    Args:
        ratings_df: Ratings dataframe
        save_path: Optional path to save matrix (as .npz)
        
    Returns:
        Tuple of (matrix, user_mapping, anime_mapping)
    """
    logger.info("Creating user-item matrix...")
    
    # Create mappings
    unique_users = ratings_df['user_id'].unique()
    unique_anime = ratings_df['anime_id'].unique()
    
    user_to_idx = {user: idx for idx, user in enumerate(unique_users)}
    anime_to_idx = {anime: idx for idx, anime in enumerate(unique_anime)}
    
    # Create matrix
    n_users = len(unique_users)
    n_anime = len(unique_anime)
    
    matrix = np.zeros((n_users, n_anime))
    
    for _, row in ratings_df.iterrows():
        user_idx = user_to_idx[row['user_id']]
        anime_idx = anime_to_idx[row['anime_id']]
        matrix[user_idx, anime_idx] = row['rating']
    
    logger.info(f"Created matrix of shape {matrix.shape}")
    logger.info(f"Sparsity: {(matrix == 0).sum() / matrix.size * 100:.2f}%")
    
    # Save if path provided
    if save_path:
        from scipy.sparse import csr_matrix
        sparse_matrix = csr_matrix(matrix)
        np.savez_compressed(save_path, 
                          data=sparse_matrix.data,
                          indices=sparse_matrix.indices,
                          indptr=sparse_matrix.indptr,
                          shape=sparse_matrix.shape)
        logger.info(f"Saved matrix to {save_path}")
    
    return matrix, user_to_idx, anime_to_idx


def get_data_statistics(anime_df: pd.DataFrame, ratings_df: Optional[pd.DataFrame] = None) -> dict:
    """
    Get comprehensive statistics about the dataset
    
    Args:
        anime_df: Anime dataframe
        ratings_df: Optional ratings dataframe
        
    Returns:
        Dictionary with statistics
    """
    stats = {
        'anime': {
            'total': len(anime_df),
            'avg_score': anime_df['Score'].mean(),
            'median_score': anime_df['Score'].median(),
            'total_genres': len(anime_df['Genres'].str.split(',').explode().unique()),
            'types': anime_df['Type'].value_counts().to_dict()
        }
    }
    
    if ratings_df is not None:
        stats['ratings'] = {
            'total': len(ratings_df),
            'unique_users': ratings_df['user_id'].nunique(),
            'unique_anime': ratings_df['anime_id'].nunique(),
            'avg_rating': ratings_df['rating'].mean(),
            'median_rating': ratings_df['rating'].median(),
            'sparsity': 1 - (len(ratings_df) / (ratings_df['user_id'].nunique() * 
                                                ratings_df['anime_id'].nunique()))
        }
    
    return stats


if __name__ == "__main__":
    from config import ANIME_RAW, RATING_RAW
    
    # Test loading
    anime_df = load_anime_data(ANIME_RAW)
    print(f"\nAnime data shape: {anime_df.shape}")
    print(f"Columns: {anime_df.columns.tolist()}")
    
    # Load small sample of ratings
    ratings_df = load_ratings_data(RATING_RAW, nrows=100000)
    print(f"\nRatings data shape: {ratings_df.shape}")
    
    # Get statistics
    stats = get_data_statistics(anime_df, ratings_df)
    print(f"\nDataset Statistics:")
    print(stats)