"""
Unit tests for recommendation models
"""

import unittest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from src.models.content_based import ContentBasedRecommender
from src.models.collaborative import CollaborativeFilteringRecommender
from src.models.hybrid_recommender import HybridRecommender


class TestContentBasedRecommender(unittest.TestCase):
    """Test Content-Based model"""
    
    def setUp(self):
        """Set up test data"""
        self.anime_df = pd.DataFrame({
            'MAL_ID': [1, 2, 3, 4, 5],
            'Name': ['Anime 1', 'Anime 2', 'Anime 3', 'Anime 4', 'Anime 5'],
            'Genres': ['Action, Comedy', 'Action, Drama', 'Comedy, Romance', 'Action, Fantasy', 'Drama, Romance'],
            'Type': ['TV', 'Movie', 'TV', 'OVA', 'TV'],
            'Score': [8.5, 7.8, 6.9, 9.1, 7.5],
            'Episodes': [12, 1, 24, 6, 13],
            'Studios': ['Studio A', 'Studio B', 'Studio A', 'Studio C', 'Studio B'],
            'Source': ['Manga', 'Original', 'Light Novel', 'Manga', 'Manga'],
            'Rating': ['PG-13', 'R', 'PG', 'PG-13', 'PG-13']
        })
        
        self.model = ContentBasedRecommender()
    
    def test_create_content_features(self):
        """Test content feature creation"""
        df = self.model.create_content_features(self.anime_df)
        
        self.assertIn('content_features', df.columns)
        self.assertFalse(df['content_features'].isna().any())
        self.assertTrue(all(isinstance(x, str) for x in df['content_features']))
    
    def test_fit(self):
        """Test model training"""
        self.model.fit(self.anime_df)
        
        # Check if model components are created
        self.assertIsNotNone(self.model.similarity_matrix)
        self.assertIsNotNone(self.model.anime_ids)
        self.assertEqual(len(self.model.anime_ids), len(self.anime_df))
    
    def test_recommend(self):
        """Test recommendation generation"""
        self.model.fit(self.anime_df)
        
        # Get recommendations for anime 1
        recommendations = self.model.recommend(1, top_n=3)
        
        # Should return recommendations
        self.assertGreater(len(recommendations), 0)
        self.assertLessEqual(len(recommendations), 3)
        
        # Should not include the input anime
        self.assertNotIn(1, recommendations['MAL_ID'].values)
        
        # Should have similarity scores
        self.assertIn('similarity_score', recommendations.columns)
    
    def test_recommend_with_filters(self):
        """Test recommendations with filters"""
        self.model.fit(self.anime_df)
        
        # Recommend with minimum score
        recommendations = self.model.recommend(1, top_n=5, min_score=8.0)
        
        # All recommendations should have score >= 8.0
        self.assertTrue((recommendations['Score'] >= 8.0).all())


class TestCollaborativeFilteringRecommender(unittest.TestCase):
    """Test Collaborative Filtering model"""
    
    def setUp(self):
        """Set up test data"""
        self.anime_df = pd.DataFrame({
            'MAL_ID': [1, 2, 3, 4, 5],
            'Name': ['Anime 1', 'Anime 2', 'Anime 3', 'Anime 4', 'Anime 5'],
            'Genres': ['Action', 'Drama', 'Comedy', 'Fantasy', 'Romance'],
            'Type': ['TV', 'Movie', 'TV', 'OVA', 'TV'],
            'Score': [8.5, 7.8, 6.9, 9.1, 7.5],
            'Episodes': [12, 1, 24, 6, 13]
        })
        
        # Create sample ratings
        self.ratings_df = pd.DataFrame({
            'user_id': [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4],
            'anime_id': [1, 2, 3, 1, 3, 4, 2, 4, 5, 1, 3, 5],
            'rating': [9, 8, 7, 10, 8, 9, 7, 9, 8, 8, 7, 9]
        })
        
        self.model = CollaborativeFilteringRecommender(algorithm='SVD')
    
    def test_prepare_data(self):
        """Test data preparation"""
        trainset, testset = self.model.prepare_data(
            self.ratings_df, 
            self.anime_df, 
            test_size=0.2
        )
        
        self.assertIsNotNone(trainset)
        self.assertIsNotNone(testset)
        self.assertGreater(trainset.n_ratings, 0)
    
    def test_fit(self):
        """Test model training"""
        trainset, testset = self.model.prepare_data(
            self.ratings_df,
            self.anime_df,
            test_size=0.2
        )
        
        self.model.fit(trainset)
        
        # Model should be trained
        self.assertIsNotNone(self.model.model)
    
    def test_predict(self):
        """Test rating prediction"""
        trainset, _ = self.model.prepare_data(
            self.ratings_df,
            self.anime_df,
            test_size=0.2
        )
        self.model.fit(trainset)
        
        # Predict rating
        predicted = self.model.predict(user_id=1, anime_id=5)
        
        # Should return a number between 1 and 10
        self.assertGreaterEqual(predicted, 1)
        self.assertLessEqual(predicted, 10)


class TestHybridRecommender(unittest.TestCase):
    """Test Hybrid model"""
    
    def setUp(self):
        """Set up test data"""
        self.anime_df = pd.DataFrame({
            'MAL_ID': [1, 2, 3, 4, 5],
            'Name': ['Anime 1', 'Anime 2', 'Anime 3', 'Anime 4', 'Anime 5'],
            'Genres': ['Action, Comedy', 'Action, Drama', 'Comedy, Romance', 'Action, Fantasy', 'Drama, Romance'],
            'Type': ['TV', 'Movie', 'TV', 'OVA', 'TV'],
            'Score': [8.5, 7.8, 6.9, 9.1, 7.5],
            'Episodes': [12, 1, 24, 6, 13],
            'Studios': ['Studio A', 'Studio B', 'Studio A', 'Studio C', 'Studio B'],
            'Source': ['Manga', 'Original', 'Light Novel', 'Manga', 'Manga'],
            'Rating': ['PG-13', 'R', 'PG', 'PG-13', 'PG-13']
        })
        
        self.ratings_df = pd.DataFrame({
            'user_id': [1, 1, 1, 2, 2, 2, 3, 3, 3],
            'anime_id': [1, 2, 3, 1, 3, 4, 2, 4, 5],
            'rating': [9, 8, 7, 10, 8, 9, 7, 9, 8]
        })
        
        # Create individual models
        self.cb_model = ContentBasedRecommender()
        self.cb_model.fit(self.anime_df)
        
        self.cf_model = CollaborativeFilteringRecommender()
        trainset, _ = self.cf_model.prepare_data(
            self.ratings_df, 
            self.anime_df,
            test_size=0.2
        )
        self.cf_model.fit(trainset)
        
        # Create hybrid model
        self.hybrid_model = HybridRecommender(
            content_model=self.cb_model,
            collaborative_model=self.cf_model,
            content_weight=0.4,
            collaborative_weight=0.6,
            strategy='weighted'
        )
    
    def test_weighted_recommendation(self):
        """Test weighted hybrid recommendation"""
        recommendations = self.hybrid_model.recommend_weighted(
            anime_id=1,
            top_n=3
        )
        
        # Should return recommendations
        self.assertGreater(len(recommendations), 0)
        self.assertLessEqual(len(recommendations), 3)
        
        # Should have hybrid scores
        if len(recommendations) > 0:
            self.assertIn('hybrid_score', recommendations.columns)
    
    def test_strategy_switching(self):
        """Test strategy changing"""
        # Test weighted strategy
        self.hybrid_model.strategy = 'weighted'
        recs1 = self.hybrid_model.recommend(anime_id=1, top_n=3)
        
        # Test mixed strategy
        self.hybrid_model.strategy = 'mixed'
        recs2 = self.hybrid_model.recommend(anime_id=1, top_n=3)
        
        # Both should return results
        self.assertGreater(len(recs1), 0)
        self.assertGreater(len(recs2), 0)


class TestModelIntegration(unittest.TestCase):
    """Integration tests for all models"""
    
    def test_end_to_end_workflow(self):
        """Test complete workflow from data to recommendations"""
        # Create sample data
        anime_df = pd.DataFrame({
            'MAL_ID': range(1, 11),
            'Name': [f'Anime {i}' for i in range(1, 11)],
            'Genres': ['Action'] * 5 + ['Drama'] * 5,
            'Type': ['TV'] * 10,
            'Score': np.random.uniform(6, 10, 10),
            'Episodes': [12] * 10,
            'Studios': ['Studio A'] * 10,
            'Source': ['Manga'] * 10,
            'Rating': ['PG-13'] * 10
        })
        
        # Train content-based model
        cb_model = ContentBasedRecommender()
        cb_model.fit(anime_df)
        
        # Get recommendations
        recommendations = cb_model.recommend(1, top_n=5)
        
        # Verify recommendations
        self.assertGreater(len(recommendations), 0)
        self.assertIn('Name', recommendations.columns)
        self.assertIn('Score', recommendations.columns)
        self.assertIn('similarity_score', recommendations.columns)


if __name__ == '__main__':
    unittest.main(verbosity=2)