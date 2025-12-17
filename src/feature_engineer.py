"""
Feature engineering for anime recommendation system
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import TruncatedSVD
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Feature engineering pipeline for anime data"""
    
    def __init__(self):
        self.tfidf_vectorizer = None
        self.scaler = None
        self.svd = None
    
    def create_text_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create combined text features from multiple columns
        
        Args:
            df: Anime dataframe
            
        Returns:
            DataFrame with text_features column
        """
        logger.info("Creating text features...")
        
        df = df.copy()
        df['text_features'] = ''
        
        # Genre (weighted 3x by repeating)
        if 'Genres' in df.columns:
            df['text_features'] += (df['Genres'].fillna('') + ' ') * 3
        
        # Type (weighted 2x)
        if 'Type' in df.columns:
            df['text_features'] += (df['Type'].fillna('') + ' ') * 2
        
        # Source
        if 'Source' in df.columns:
            df['text_features'] += df['Source'].fillna('') + ' '
        
        # Studios (weighted 2x)
        if 'Studios' in df.columns:
            df['text_features'] += (df['Studios'].fillna('') + ' ') * 2
        
        # Producers
        if 'Producers' in df.columns:
            df['text_features'] += df['Producers'].fillna('') + ' '
        
        # Rating
        if 'Rating' in df.columns:
            df['text_features'] += df['Rating'].fillna('') + ' '
        
        # Premiered (season info)
        if 'Premiered' in df.columns:
            df['text_features'] += df['Premiered'].fillna('') + ' '
        
        # Clean up
        df['text_features'] = df['text_features'].str.lower().str.strip()
        df['text_features'] = df['text_features'].str.replace(r'[^\w\s]', '', regex=True)
        
        logger.info(f"Created text features for {len(df)} anime")
        return df
    
    def tfidf_vectorize(self, texts, max_features=5000, ngram_range=(1, 2)):
        """
        Apply TF-IDF vectorization to text features
        
        Args:
            texts: List or Series of text
            max_features: Maximum number of features
            ngram_range: N-gram range
            
        Returns:
            TF-IDF matrix
        """
        logger.info(f"Applying TF-IDF vectorization (max_features={max_features})...")
        
        if self.tfidf_vectorizer is None:
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=max_features,
                ngram_range=ngram_range,
                stop_words='english',
                min_df=2,
                max_df=0.8
            )
            tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts)
        else:
            tfidf_matrix = self.tfidf_vectorizer.transform(texts)
        
        logger.info(f"TF-IDF matrix shape: {tfidf_matrix.shape}")
        return tfidf_matrix
    
    def create_numeric_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create and scale numeric features
        
        Args:
            df: Anime dataframe
            
        Returns:
            DataFrame with scaled numeric features
        """
        logger.info("Creating numeric features...")
        
        df = df.copy()
        
        # Episodes (log transform to handle outliers)
        if 'Episodes' in df.columns:
            df['Episodes_log'] = np.log1p(df['Episodes'].fillna(0))
        
        # Members (log transform)
        if 'Members' in df.columns:
            df['Members_log'] = np.log1p(df['Members'].fillna(0))
        
        # Favorites (log transform)
        if 'Favorites' in df.columns:
            df['Favorites_log'] = np.log1p(df['Favorites'].fillna(0))
        
        # Scored By (log transform)
        if 'Scored By' in df.columns:
            df['Scored_By_log'] = np.log1p(df['Scored By'].fillna(0))
        
        # Popularity rank (inverse - lower is better)
        if 'Popularity' in df.columns:
            max_pop = df['Popularity'].max()
            df['Popularity_normalized'] = 1 - (df['Popularity'].fillna(max_pop) / max_pop)
        
        # Score
        if 'Score' in df.columns:
            df['Score_normalized'] = df['Score'].fillna(0) / 10.0
        
        # Rank (inverse)
        if 'Rank' in df.columns:
            max_rank = df['Rank'].max()
            df['Rank_normalized'] = 1 - (df['Rank'].fillna(max_rank) / max_rank)
        
        logger.info("Numeric features created")
        return df
    
    def create_genre_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create one-hot encoded genre features
        
        Args:
            df: Anime dataframe
            
        Returns:
            DataFrame with genre columns
        """
        logger.info("Creating genre features...")
        
        df = df.copy()
        
        # Get all unique genres
        all_genres = set()
        for genres in df['Genres'].dropna():
            all_genres.update([g.strip() for g in str(genres).split(',')])
        
        all_genres = sorted(list(all_genres))
        logger.info(f"Found {len(all_genres)} unique genres")
        
        # Create binary columns for each genre
        for genre in all_genres:
            df[f'genre_{genre}'] = df['Genres'].str.contains(genre, case=False, na=False).astype(int)
        
        return df
    
    def create_embeddings(self, tfidf_matrix, n_components=100):
        """
        Create lower-dimensional embeddings using SVD
        
        Args:
            tfidf_matrix: TF-IDF sparse matrix
            n_components: Number of components for SVD
            
        Returns:
            Dense embedding matrix
        """
        logger.info(f"Creating embeddings with {n_components} components...")
        
        if self.svd is None:
            self.svd = TruncatedSVD(n_components=n_components, random_state=42)
            embeddings = self.svd.fit_transform(tfidf_matrix)
        else:
            embeddings = self.svd.transform(tfidf_matrix)
        
        logger.info(f"Embeddings shape: {embeddings.shape}")
        logger.info(f"Explained variance: {self.svd.explained_variance_ratio_.sum():.3f}")
        
        return embeddings
    
    def build_similarity_matrix(self, features, metric='cosine'):
        """
        Build similarity matrix from features
        
        Args:
            features: Feature matrix
            metric: Similarity metric ('cosine', 'euclidean', etc.)
            
        Returns:
            Similarity matrix
        """
        logger.info(f"Building similarity matrix using {metric} metric...")
        
        from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
        
        if metric == 'cosine':
            similarity = cosine_similarity(features)
        elif metric == 'euclidean':
            distances = euclidean_distances(features)
            # Convert distances to similarities
            similarity = 1 / (1 + distances)
        else:
            raise ValueError(f"Unknown metric: {metric}")
        
        logger.info(f"Similarity matrix shape: {similarity.shape}")
        return similarity
    
    def create_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create time-based features from Premiered/Aired columns
        
        Args:
            df: Anime dataframe
            
        Returns:
            DataFrame with time features
        """
        logger.info("Creating time features...")
        
        df = df.copy()
        
        # Extract season from Premiered (e.g., "Spring 2020")
        if 'Premiered' in df.columns:
            df['Season'] = df['Premiered'].str.extract(r'(Spring|Summer|Fall|Winter)', expand=False)
            df['Year'] = df['Premiered'].str.extract(r'(\d{4})', expand=False).astype(float)
            
            # Encode seasons as numbers
            season_map = {'Winter': 1, 'Spring': 2, 'Summer': 3, 'Fall': 4}
            df['Season_numeric'] = df['Season'].map(season_map)
        
        return df
    
    def create_all_features(self, df: pd.DataFrame, 
                          include_text=True,
                          include_numeric=True,
                          include_genre=True,
                          include_time=True) -> pd.DataFrame:
        """
        Create all features in one pipeline
        
        Args:
            df: Anime dataframe
            include_text: Include text features
            include_numeric: Include numeric features
            include_genre: Include genre features
            include_time: Include time features
            
        Returns:
            DataFrame with all features
        """
        logger.info("Creating all features...")
        
        result_df = df.copy()
        
        if include_text:
            result_df = self.create_text_features(result_df)
        
        if include_numeric:
            result_df = self.create_numeric_features(result_df)
        
        if include_genre:
            result_df = self.create_genre_features(result_df)
        
        if include_time:
            result_df = self.create_time_features(result_df)
        
        logger.info(f"Feature engineering complete. Shape: {result_df.shape}")
        return result_df


if __name__ == "__main__":
    from config import ANIME_CLEANED
    
    # Load data
    anime_df = pd.read_csv(ANIME_CLEANED)
    
    # Initialize feature engineer
    fe = FeatureEngineer()
    
    # Create all features
    anime_features = fe.create_all_features(anime_df)
    print(f"\nFeature engineered data shape: {anime_features.shape}")
    print(f"New columns: {[col for col in anime_features.columns if col not in anime_df.columns]}")