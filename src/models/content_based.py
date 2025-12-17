import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import joblib
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ContentBasedRecommender:
    """Content-based recommendation system using TF-IDF and cosine similarity"""
    
    def __init__(self, max_features=5000, ngram_range=(1, 2)):
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            stop_words='english'
        )
        self.similarity_matrix = None
        self.anime_df = None
        self.anime_ids = None
        self.id_to_idx = None
        self.idx_to_id = None
    
    def create_content_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create combined content features from anime metadata"""
        logger.info("Creating content features...")
        
        # Combine multiple text features
        df['content_features'] = ''
        
        # Add genres (weighted more by repeating)
        if 'Genres' in df.columns:
            df['content_features'] += df['Genres'].fillna('') + ' ' + df['Genres'].fillna('')
        
        # Add type
        if 'Type' in df.columns:
            df['content_features'] += ' ' + df['Type'].fillna('')
        
        # Add source
        if 'Source' in df.columns:
            df['content_features'] += ' ' + df['Source'].fillna('')
        
        # Add studios
        if 'Studios' in df.columns:
            df['content_features'] += ' ' + df['Studios'].fillna('')
        
        # Add rating category
        if 'Rating' in df.columns:
            df['content_features'] += ' ' + df['Rating'].fillna('')
        
        # Clean up
        df['content_features'] = df['content_features'].str.lower().str.strip()
        
        return df
    
    def fit(self, anime_df: pd.DataFrame):
        """Train the content-based model"""
        logger.info("Training content-based model...")
        
        self.anime_df = anime_df.copy()
        
        # Create content features
        self.anime_df = self.create_content_features(self.anime_df)
        
        # Create TF-IDF matrix
        tfidf_matrix = self.vectorizer.fit_transform(self.anime_df['content_features'])
        logger.info(f"TF-IDF matrix shape: {tfidf_matrix.shape}")
        
        # Compute cosine similarity
        self.similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)
        logger.info("Similarity matrix computed")
        
        # Create mapping dictionaries
        self.anime_ids = self.anime_df['MAL_ID'].values
        self.id_to_idx = {anime_id: idx for idx, anime_id in enumerate(self.anime_ids)}
        self.idx_to_id = {idx: anime_id for idx, anime_id in enumerate(self.anime_ids)}
        
        logger.info("Content-based model trained successfully")
    
    def recommend(self, anime_id: int, top_n: int = 10, 
                 min_score: float = 0.0, exclude_same_id: bool = True) -> pd.DataFrame:
        """
        Get recommendations for a given anime
        """
        if anime_id not in self.id_to_idx:
            logger.warning(f"Anime ID {anime_id} not found")
            return pd.DataFrame()
        
        # Lấy index của anime đầu vào
        idx = self.id_to_idx[anime_id]
        
        # Lấy mảng độ tương đồng
        sim_scores = self.similarity_matrix[idx]
        
        # Sắp xếp index theo độ tương đồng giảm dần
        sorted_indices = np.argsort(sim_scores)[::-1]
        
        # Lấy danh sách ứng viên (lấy dư ra top_n * 5 để trừ hao lúc lọc)
        candidate_indices = sorted_indices[:top_n * 5]
        
        # Tạo dataframe kết quả từ danh sách ứng viên
        result = self.anime_df.iloc[candidate_indices].copy()
        result['similarity_score'] = sim_scores[candidate_indices]
        
        # --- ÁP DỤNG BỘ LỌC (PHẦN QUAN TRỌNG ĐỂ FIX LỖI) ---
        
        # 1. Loại bỏ chính anime đang query
        if exclude_same_id:
            result = result[result['MAL_ID'] != anime_id]
        
        # 2. Lọc theo điểm số tối thiểu (Loại bỏ hẳn các dòng không đạt)
        if min_score > 0 and 'Score' in result.columns:
            result = result[result['Score'] >= min_score]
            
        # ---------------------------------------------------
        
        # Trả về top N dòng đầu tiên sau khi đã lọc sạch
        cols = ['MAL_ID', 'Name', 'Genres', 'Score', 'Type', 'Episodes', 'similarity_score']
        return result.head(top_n)[cols]
    
    def batch_recommend(self, anime_ids: list, top_n: int = 10) -> dict:
        """Get recommendations for multiple anime"""
        results = {}
        for anime_id in anime_ids:
            results[anime_id] = self.recommend(anime_id, top_n)
        return results
    
    def save_model(self, model_path: str, vectorizer_path: str):
        """Save the trained model"""
        model_data = {
            'similarity_matrix': self.similarity_matrix,
            'anime_df': self.anime_df,
            'anime_ids': self.anime_ids,
            'id_to_idx': self.id_to_idx,
            'idx_to_id': self.idx_to_id
        }
        joblib.dump(model_data, model_path)
        joblib.dump(self.vectorizer, vectorizer_path)
        logger.info(f"Model saved to {model_path}")
    
    def load_model(self, model_path: str, vectorizer_path: str):
        """Load a trained model"""
        model_data = joblib.load(model_path)
        self.similarity_matrix = model_data['similarity_matrix']
        self.anime_df = model_data['anime_df']
        self.anime_ids = model_data['anime_ids']
        self.id_to_idx = model_data['id_to_idx']
        self.idx_to_id = model_data['idx_to_id']
        self.vectorizer = joblib.load(vectorizer_path)
        logger.info(f"Model loaded from {model_path}")


if __name__ == "__main__":
    # Example usage
    from config import ANIME_CLEANED, CONTENT_MODEL, TFIDF_VECTORIZER
    
    # Load data
    anime_df = pd.read_csv(ANIME_CLEANED)
    
    # Train model
    recommender = ContentBasedRecommender()
    recommender.fit(anime_df)
    
    # Get recommendations for Death Note (MAL_ID: 1535)
    recommendations = recommender.recommend(1535, top_n=10)
    print("\nRecommendations for Death Note:")
    print(recommendations)
    
    # Save model
    recommender.save_model(CONTENT_MODEL, TFIDF_VECTORIZER)