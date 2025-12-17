"""
Hybrid Recommendation System combining Content-Based and Collaborative Filtering
"""

import pandas as pd
import numpy as np
import joblib
import logging
from typing import Optional, Dict, List

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HybridRecommender:
    """
    Hybrid recommender combining content-based and collaborative filtering
    
    Strategies:
    1. Weighted: Combine scores with weights
    2. Switching: Use CB for cold-start, CF for warm-start
    3. Mixed: Mix recommendations from both systems
    """
    
    def __init__(self, 
                 content_model=None, 
                 collaborative_model=None,
                 content_weight=0.4,
                 collaborative_weight=0.6,
                 strategy='weighted'):
        """
        Initialize hybrid recommender
        
        Args:
            content_model: Content-based recommender
            collaborative_model: Collaborative filtering recommender
            content_weight: Weight for content-based scores (0-1)
            collaborative_weight: Weight for collaborative scores (0-1)
            strategy: 'weighted', 'switching', or 'mixed'
        """
        self.content_model = content_model
        self.collaborative_model = collaborative_model
        self.content_weight = content_weight
        self.collaborative_weight = collaborative_weight
        self.strategy = strategy
        
        # Normalize weights
        total = content_weight + collaborative_weight
        self.content_weight = content_weight / total
        self.collaborative_weight = collaborative_weight / total
        
        logger.info(f"Hybrid Recommender initialized with strategy: {strategy}")
        logger.info(f"Weights - Content: {self.content_weight:.2f}, Collaborative: {self.collaborative_weight:.2f}")
    
    def recommend_weighted(self, anime_id: int = None, user_id: int = None,
                          top_n: int = 10, min_score: float = 0.0) -> pd.DataFrame:
        """
        Weighted hybrid recommendation
        
        Combines scores from both models using weights
        
        Args:
            anime_id: Anime ID for content-based (optional)
            user_id: User ID for collaborative (optional)
            top_n: Number of recommendations
            min_score: Minimum anime score
            
        Returns:
            DataFrame with recommendations
        """
        if anime_id is None and user_id is None:
            raise ValueError("Either anime_id or user_id must be provided")
        
        recommendations = {}
        
        # Get content-based recommendations
        if anime_id is not None and self.content_model is not None:
            try:
                cb_recs = self.content_model.recommend(
                    anime_id, 
                    top_n=top_n * 2,  # Get more to merge
                    min_score=min_score
                )
                
                for _, row in cb_recs.iterrows():
                    mal_id = row['MAL_ID']
                    score = row.get('similarity_score', 0) * self.content_weight
                    
                    if mal_id not in recommendations:
                        recommendations[mal_id] = {
                            'anime_data': row,
                            'cb_score': row.get('similarity_score', 0),
                            'cf_score': 0,
                            'hybrid_score': score
                        }
                    else:
                        recommendations[mal_id]['cb_score'] = row.get('similarity_score', 0)
                        recommendations[mal_id]['hybrid_score'] += score
                
                logger.info(f"Got {len(cb_recs)} content-based recommendations")
            except Exception as e:
                logger.warning(f"Content-based recommendation failed: {str(e)}")
        
        # Get collaborative filtering recommendations
        if user_id is not None and self.collaborative_model is not None:
            try:
                cf_recs = self.collaborative_model.recommend_for_user(
                    user_id,
                    top_n=top_n * 2,
                    min_score=min_score
                )
                
                # Normalize predicted ratings to 0-1 scale
                if len(cf_recs) > 0:
                    max_rating = cf_recs['predicted_rating'].max()
                    min_rating = cf_recs['predicted_rating'].min()
                    rating_range = max_rating - min_rating if max_rating > min_rating else 1
                
                for _, row in cf_recs.iterrows():
                    mal_id = row['MAL_ID']
                    
                    # Normalize rating
                    normalized_rating = (row['predicted_rating'] - min_rating) / rating_range
                    score = normalized_rating * self.collaborative_weight
                    
                    if mal_id not in recommendations:
                        recommendations[mal_id] = {
                            'anime_data': row,
                            'cb_score': 0,
                            'cf_score': normalized_rating,
                            'hybrid_score': score
                        }
                    else:
                        recommendations[mal_id]['cf_score'] = normalized_rating
                        recommendations[mal_id]['hybrid_score'] += score
                
                logger.info(f"Got {len(cf_recs)} collaborative recommendations")
            except Exception as e:
                logger.warning(f"Collaborative recommendation failed: {str(e)}")
        
        if not recommendations:
            logger.warning("No recommendations generated")
            return pd.DataFrame()
        
        # Sort by hybrid score
        sorted_recs = sorted(
            recommendations.items(),
            key=lambda x: x[1]['hybrid_score'],
            reverse=True
        )[:top_n]
        
        # Create result dataframe
        result_data = []
        for mal_id, rec_data in sorted_recs:
            anime_data = rec_data['anime_data']
            result_data.append({
                'MAL_ID': mal_id,
                'Name': anime_data['Name'],
                'Genres': anime_data.get('Genres', ''),
                'Score': anime_data.get('Score', 0),
                'Type': anime_data.get('Type', ''),
                'Episodes': anime_data.get('Episodes', 0),
                'cb_score': rec_data['cb_score'],
                'cf_score': rec_data['cf_score'],
                'hybrid_score': rec_data['hybrid_score']
            })
        
        result_df = pd.DataFrame(result_data)
        logger.info(f"Generated {len(result_df)} hybrid recommendations")
        
        return result_df
    
    def recommend_switching(self, anime_id: int = None, user_id: int = None,
                           top_n: int = 10, min_score: float = 0.0,
                           user_rating_threshold: int = 10) -> pd.DataFrame:
        """
        Switching hybrid recommendation
        
        Uses content-based for cold-start (new users/items),
        collaborative for warm-start (known users)
        
        Args:
            anime_id: Anime ID for content-based
            user_id: User ID for collaborative
            top_n: Number of recommendations
            min_score: Minimum anime score
            user_rating_threshold: Threshold to switch from CB to CF
            
        Returns:
            DataFrame with recommendations
        """
        # Check if user is cold-start
        is_cold_start = True
        
        if user_id is not None and self.collaborative_model is not None:
            try:
                # Check if user has enough ratings
                trainset = self.collaborative_model.trainset
                user_ratings = [r for u, i, r in trainset.all_ratings() if u == user_id]
                
                if len(user_ratings) >= user_rating_threshold:
                    is_cold_start = False
            except:
                pass
        
        # Use appropriate model
        if is_cold_start or anime_id is not None:
            logger.info("Using content-based (cold-start or anime-based)")
            if self.content_model is None:
                return pd.DataFrame()
            
            return self.content_model.recommend(anime_id, top_n, min_score)
        else:
            logger.info("Using collaborative filtering (warm-start)")
            if self.collaborative_model is None:
                return pd.DataFrame()
            
            return self.collaborative_model.recommend_for_user(user_id, top_n, min_score)
    
    def recommend_mixed(self, anime_id: int = None, user_id: int = None,
                       top_n: int = 10, min_score: float = 0.0,
                       mix_ratio: float = 0.5) -> pd.DataFrame:
        """
        Mixed hybrid recommendation
        
        Takes N recommendations from each system and mixes them
        
        Args:
            anime_id: Anime ID for content-based
            user_id: User ID for collaborative
            top_n: Number of recommendations
            min_score: Minimum anime score
            mix_ratio: Ratio of CB recommendations (0-1)
            
        Returns:
            DataFrame with recommendations
        """
        n_cb = int(top_n * mix_ratio)
        n_cf = top_n - n_cb
        
        recommendations = []
        seen_ids = set()
        
        # Get content-based recommendations
        if anime_id is not None and self.content_model is not None and n_cb > 0:
            try:
                cb_recs = self.content_model.recommend(anime_id, n_cb, min_score)
                for _, row in cb_recs.iterrows():
                    if row['MAL_ID'] not in seen_ids:
                        recommendations.append(row)
                        seen_ids.add(row['MAL_ID'])
                
                logger.info(f"Added {len(cb_recs)} CB recommendations")
            except Exception as e:
                logger.warning(f"CB recommendation failed: {str(e)}")
        
        # Get collaborative recommendations
        if user_id is not None and self.collaborative_model is not None and n_cf > 0:
            try:
                cf_recs = self.collaborative_model.recommend_for_user(user_id, n_cf * 2, min_score)
                
                added = 0
                for _, row in cf_recs.iterrows():
                    if row['MAL_ID'] not in seen_ids and added < n_cf:
                        recommendations.append(row)
                        seen_ids.add(row['MAL_ID'])
                        added += 1
                
                logger.info(f"Added {added} CF recommendations")
            except Exception as e:
                logger.warning(f"CF recommendation failed: {str(e)}")
        
        if not recommendations:
            return pd.DataFrame()
        
        result_df = pd.DataFrame(recommendations)
        logger.info(f"Mixed {len(result_df)} recommendations")
        
        return result_df.head(top_n)
    
    def recommend(self, anime_id: int = None, user_id: int = None,
                 top_n: int = 10, min_score: float = 0.0, **kwargs) -> pd.DataFrame:
        """
        Get recommendations using configured strategy
        
        Args:
            anime_id: Anime ID for content-based
            user_id: User ID for collaborative
            top_n: Number of recommendations
            min_score: Minimum anime score
            **kwargs: Additional arguments for specific strategies
            
        Returns:
            DataFrame with recommendations
        """
        if self.strategy == 'weighted':
            return self.recommend_weighted(anime_id, user_id, top_n, min_score)
        elif self.strategy == 'switching':
            return self.recommend_switching(anime_id, user_id, top_n, min_score, **kwargs)
        elif self.strategy == 'mixed':
            return self.recommend_mixed(anime_id, user_id, top_n, min_score, **kwargs)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")
    
    def save_model(self, model_path: str):
        """Save hybrid model"""
        model_data = {
            'content_model': self.content_model,
            'collaborative_model': self.collaborative_model,
            'content_weight': self.content_weight,
            'collaborative_weight': self.collaborative_weight,
            'strategy': self.strategy
        }
        joblib.dump(model_data, model_path)
        logger.info(f"Hybrid model saved to {model_path}")
    
    def load_model(self, model_path: str):
        """Load hybrid model"""
        model_data = joblib.load(model_path)
        self.content_model = model_data['content_model']
        self.collaborative_model = model_data['collaborative_model']
        self.content_weight = model_data['content_weight']
        self.collaborative_weight = model_data['collaborative_weight']
        self.strategy = model_data['strategy']
        logger.info(f"Hybrid model loaded from {model_path}")


if __name__ == "__main__":
    from config import CONTENT_MODEL, COLLABORATIVE_MODEL, TFIDF_VECTORIZER, HYBRID_MODEL
    from content_based import ContentBasedRecommender
    from collaborative import CollaborativeFilteringRecommender
    
    # Load individual models
    cb_model = ContentBasedRecommender()
    cb_model.load_model(CONTENT_MODEL, TFIDF_VECTORIZER)
    
    cf_model = CollaborativeFilteringRecommender()
    cf_model.load_model(COLLABORATIVE_MODEL)
    
    # Create hybrid model
    hybrid = HybridRecommender(
        content_model=cb_model,
        collaborative_model=cf_model,
        content_weight=0.4,
        collaborative_weight=0.6,
        strategy='weighted'
    )
    
    # Test recommendations
    print("\n=== Testing Weighted Hybrid ===")
    recs = hybrid.recommend(anime_id=1535, top_n=10)  # Death Note
    print(recs[['Name', 'Score', 'cb_score', 'cf_score', 'hybrid_score']])
    
    # Save hybrid model
    hybrid.save_model(HYBRID_MODEL)
    print(f"\nHybrid model saved to {HYBRID_MODEL}")