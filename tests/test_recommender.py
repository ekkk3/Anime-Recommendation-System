"""
Integration tests for recommendation system
"""

import unittest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from src.models.content_based import ContentBasedRecommender
from src.evaluator import ModelEvaluator


class TestRecommendationQuality(unittest.TestCase):
    """Test recommendation quality and diversity"""
    
    def setUp(self):
        """Set up test environment"""
        # Create diverse anime dataset
        self.anime_df = pd.DataFrame({
            'MAL_ID': range(1, 21),
            'Name': [f'Anime {i}' for i in range(1, 21)],
            'Genres': [
                'Action, Adventure',
                'Action, Fantasy',
                'Comedy, Slice of Life',
                'Drama, Romance',
                'Horror, Thriller',
                'Action, Sci-Fi',
                'Comedy, Romance',
                'Drama, Mystery',
                'Action, Sports',
                'Fantasy, Magic',
                'Comedy, School',
                'Drama, Historical',
                'Action, Mecha',
                'Romance, School',
                'Horror, Supernatural',
                'Adventure, Fantasy',
                'Comedy, Music',
                'Drama, Psychological',
                'Action, Military',
                'Romance, Drama'
            ],
            'Type': ['TV'] * 20,
            'Score': np.random.uniform(6.5, 9.5, 20),
            'Episodes': [12, 24, 13, 26, 12, 50, 13, 24, 25, 12, 12, 24, 26, 13, 12, 24, 12, 13, 24, 12],
            'Studios': [f'Studio {chr(65+i%5)}' for i in range(20)],
            'Source': ['Manga', 'Light Novel', 'Original'] * 6 + ['Manga', 'Game'],
            'Rating': ['PG-13'] * 20
        })
        
        self.model = ContentBasedRecommender()
        self.model.fit(self.anime_df)
    
    def test_recommendation_diversity(self):
        """Test that recommendations are diverse"""
        recommendations = self.model.recommend(1, top_n=10)
        
        # Check diversity in genres
        all_genres = set()
        for genres in recommendations['Genres']:
            all_genres.update(genres.split(', '))
        
        # Should have multiple different genres
        self.assertGreater(len(all_genres), 3)
    
    def test_recommendation_relevance(self):
        """Test that recommendations are relevant to input"""
        # Get anime with 'Action' genre
        action_anime_id = 1  # Has 'Action, Adventure'
        
        recommendations = self.model.recommend(action_anime_id, top_n=5)
        
        # At least some recommendations should have 'Action'
        action_count = sum(1 for genres in recommendations['Genres'] 
                          if 'Action' in genres)
        
        self.assertGreater(action_count, 0)
    
    def test_no_duplicate_recommendations(self):
        """Test that there are no duplicate recommendations"""
        recommendations = self.model.recommend(1, top_n=10)
        
        # Check for unique MAL_IDs
        self.assertEqual(len(recommendations), recommendations['MAL_ID'].nunique())
    
    def test_score_filtering(self):
        """Test minimum score filtering"""
        min_score = 8.0
        recommendations = self.model.recommend(1, top_n=10, min_score=min_score)
        
        # All recommendations should have score >= min_score
        if len(recommendations) > 0:
            self.assertTrue((recommendations['Score'] >= min_score).all())
    
    def test_similarity_scores_ordering(self):
        """Test that recommendations are ordered by similarity"""
        recommendations = self.model.recommend(1, top_n=10)
        
        if len(recommendations) > 1:
            # Similarity scores should be in descending order
            similarity_scores = recommendations['similarity_score'].values
            self.assertTrue(all(similarity_scores[i] >= similarity_scores[i+1] 
                              for i in range(len(similarity_scores)-1)))


class TestRecommendationConsistency(unittest.TestCase):
    """Test recommendation consistency"""
    
    def setUp(self):
        """Set up test data"""
        self.anime_df = pd.DataFrame({
            'MAL_ID': range(1, 11),
            'Name': [f'Anime {i}' for i in range(1, 11)],
            'Genres': ['Action'] * 5 + ['Drama'] * 5,
            'Type': ['TV'] * 10,
            'Score': [8.0 + i*0.1 for i in range(10)],
            'Episodes': [12] * 10,
            'Studios': ['Studio A'] * 10,
            'Source': ['Manga'] * 10,
            'Rating': ['PG-13'] * 10
        })
        
        self.model = ContentBasedRecommender()
        self.model.fit(self.anime_df)
    
    def test_consistent_recommendations(self):
        """Test that same input gives same recommendations"""
        recs1 = self.model.recommend(1, top_n=5)
        recs2 = self.model.recommend(1, top_n=5)
        
        # Should return same recommendations
        pd.testing.assert_frame_equal(
            recs1.reset_index(drop=True),
            recs2.reset_index(drop=True)
        )
    
    def test_different_inputs_different_outputs(self):
        """Test that different inputs give different recommendations"""
        recs1 = self.model.recommend(1, top_n=5)
        recs2 = self.model.recommend(6, top_n=5)  # Different genre
        
        # Should have different recommendations
        ids1 = set(recs1['MAL_ID'].values)
        ids2 = set(recs2['MAL_ID'].values)
        
        # Should have some difference
        self.assertNotEqual(ids1, ids2)


class TestEvaluationMetrics(unittest.TestCase):
    """Test evaluation metrics"""
    
    def setUp(self):
        """Set up test data"""
        self.evaluator = ModelEvaluator()
    
    def test_rmse_calculation(self):
        """Test RMSE calculation"""
        true_ratings = np.array([5, 4, 3, 5, 4])
        predicted_ratings = np.array([4.5, 4.2, 3.1, 4.8, 3.9])
        
        rmse = self.evaluator.calculate_rmse(true_ratings, predicted_ratings)
        
        # RMSE should be positive
        self.assertGreater(rmse, 0)
        
        # RMSE should be reasonable
        self.assertLess(rmse, 2)
    
    def test_mae_calculation(self):
        """Test MAE calculation"""
        true_ratings = np.array([5, 4, 3, 5, 4])
        predicted_ratings = np.array([4.5, 4.2, 3.1, 4.8, 3.9])
        
        mae = self.evaluator.calculate_mae(true_ratings, predicted_ratings)
        
        # MAE should be positive
        self.assertGreater(mae, 0)
        
        # MAE should be less than max error
        self.assertLess(mae, np.max(np.abs(true_ratings - predicted_ratings)))
    
    def test_precision_at_k(self):
        """Test Precision@K"""
        recommended = [1, 2, 3, 4, 5]
        relevant = [2, 4, 6, 8]
        
        precision = self.evaluator.precision_at_k(recommended, relevant, k=5)
        
        # Should be 2/5 = 0.4 (items 2 and 4 are relevant)
        self.assertAlmostEqual(precision, 0.4)
    
    def test_recall_at_k(self):
        """Test Recall@K"""
        recommended = [1, 2, 3, 4, 5]
        relevant = [2, 4, 6, 8]
        
        recall = self.evaluator.recall_at_k(recommended, relevant, k=5)
        
        # Should be 2/4 = 0.5 (found 2 out of 4 relevant items)
        self.assertAlmostEqual(recall, 0.5)
    
    def test_f1_score_at_k(self):
        """Test F1-Score@K"""
        recommended = [1, 2, 3, 4, 5]
        relevant = [2, 4, 6, 8]
        
        f1 = self.evaluator.f1_score_at_k(recommended, relevant, k=5)
        
        # F1 should be harmonic mean of precision and recall
        precision = 0.4
        recall = 0.5
        expected_f1 = 2 * (precision * recall) / (precision + recall)
        
        self.assertAlmostEqual(f1, expected_f1)


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error handling"""
    
    def test_empty_recommendations(self):
        """Test handling of empty recommendation list"""
        anime_df = pd.DataFrame({
            'MAL_ID': [1],
            'Name': ['Anime 1'],
            'Genres': ['Action'],
            'Type': ['TV'],
            'Score': [8.0],
            'Episodes': [12],
            'Studios': ['Studio A'],
            'Source': ['Manga'],
            'Rating': ['PG-13']
        })
        
        model = ContentBasedRecommender()
        model.fit(anime_df)
        
        # Try to get recommendations (should return empty since only 1 anime)
        recommendations = model.recommend(1, top_n=5, exclude_same_id=True)
        
        # Should handle gracefully
        self.assertEqual(len(recommendations), 0)
    
    def test_invalid_anime_id(self):
        """Test handling of invalid anime ID"""
        anime_df = pd.DataFrame({
            'MAL_ID': [1, 2, 3],
            'Name': ['Anime 1', 'Anime 2', 'Anime 3'],
            'Genres': ['Action'] * 3,
            'Type': ['TV'] * 3,
            'Score': [8.0] * 3,
            'Episodes': [12] * 3,
            'Studios': ['Studio A'] * 3,
            'Source': ['Manga'] * 3,
            'Rating': ['PG-13'] * 3
        })
        
        model = ContentBasedRecommender()
        model.fit(anime_df)
        
        # Try with non-existent ID
        recommendations = model.recommend(999, top_n=5)
        
        # Should return empty dataframe
        self.assertEqual(len(recommendations), 0)
    
    def test_zero_ratings(self):
        """Test evaluation with zero predictions"""
        evaluator = ModelEvaluator()
        
        # Test with identical predictions
        true_ratings = np.array([5, 5, 5])
        predicted_ratings = np.array([5, 5, 5])
        
        rmse = evaluator.calculate_rmse(true_ratings, predicted_ratings)
        
        # RMSE should be 0
        self.assertAlmostEqual(rmse, 0)


if __name__ == '__main__':
    # Run all tests
    unittest.main(verbosity=2)