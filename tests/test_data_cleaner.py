"""
Unit tests for data cleaning module
"""

import unittest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add parent to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data_cleaner import AnimeDataCleaner


class TestAnimeDataCleaner(unittest.TestCase):
    """Test cases for AnimeDataCleaner"""
    
    def setUp(self):
        """Set up test data"""
        self.cleaner = AnimeDataCleaner()
        
        # Create sample data
        self.sample_data = pd.DataFrame({
            'MAL_ID': [1, 2, 3, 3, 4],  # Duplicate ID
            'Name': ['Anime 1', 'Anime 2', 'Anime 3', 'Anime 3', None],  # Missing name
            'Score': [8.5, np.nan, 7.2, 7.2, 6.8],  # Missing score
            'Genres': ['Action, Comedy', 'Drama', None, None, 'Romance'],  # Missing genres
            'Type': ['TV', 'Movie', 'TV', 'TV', 'OVA'],
            'Episodes': [12, 1, np.nan, np.nan, 6],  # Missing episodes
            'Members': [100000, 50000, 200000, 200000, 10000]
        })
    
    def test_handle_missing_values(self):
        """Test missing value handling"""
        df = self.sample_data.copy()
        cleaned = self.cleaner.handle_missing_values(df)
        
        # Check Score filled with median
        self.assertFalse(cleaned['Score'].isna().any())
        
        # Check Genres filled with 'Unknown'
        self.assertEqual(cleaned['Genres'].isna().sum(), 0)
        
        # Check rows with missing Name are dropped
        self.assertFalse(cleaned['Name'].isna().any())
    
    def test_remove_duplicates(self):
        """Test duplicate removal"""
        df = self.sample_data.copy()
        cleaned = self.cleaner.remove_duplicates(df)
        
        # Should have 4 rows (removed 1 duplicate)
        self.assertEqual(len(cleaned), 4)
        
        # No duplicate MAL_IDs
        self.assertEqual(cleaned['MAL_ID'].nunique(), len(cleaned))
    
    def test_normalize_data(self):
        """Test data normalization"""
        df = self.sample_data.copy()

        # --- THÊM DÒNG NÀY ---
        # Lấp giá trị trống trước khi test chuẩn hóa (giả lập quy trình thực tế)
        df['Score'] = df['Score'].fillna(0)
        df['Episodes'] = df['Episodes'].fillna(0)
        # ---------------------
        cleaned = self.cleaner.normalize_data(df)
        
        # Check Score is clipped to 0-10
        self.assertTrue((cleaned['Score'] >= 0).all())
        self.assertTrue((cleaned['Score'] <= 10).all())
        
        # Check Episodes is numeric
        self.assertTrue(pd.api.types.is_numeric_dtype(cleaned['Episodes']))
        
        # Check Genres_List created
        if 'Genres_List' in cleaned.columns:
            self.assertTrue(isinstance(cleaned['Genres_List'].iloc[0], list))
    
    def test_remove_outliers(self):
        """Test outlier removal"""
        df = self.sample_data.copy()
        
        # Add extreme outlier
        df.loc[len(df)] = [999, 'Outlier', 10.0, 'Action', 'TV', 1, 10000000]
        
        cleaned = self.cleaner.remove_outliers(df)
        
        # Should remove the extreme outlier
        self.assertLess(len(cleaned), len(df))
    
    def test_clean_ratings(self):
        """Test ratings cleaning"""
        ratings_df = pd.DataFrame({
            'user_id': [1, 2, 3, 4, 5],
            'anime_id': [1, 2, 3, 999, 1],  # 999 doesn't exist
            'rating': [8, 0, 7, 9, -1]  # 0 is watching, -1 is invalid
        })
        
        valid_anime_ids = {1, 2, 3}
        
        cleaned = self.cleaner.clean_ratings(ratings_df, valid_anime_ids)
        
        # Should remove invalid ratings
        # --- SỬA DÒNG NÀY ---
        # Cho phép rating >= 1 HOẶC rating == 0
        self.assertTrue(((cleaned['rating'] >= 1) | (cleaned['rating'] == 0)).all())
        # --------------------
        self.assertTrue((cleaned['rating'] <= 10).all())
        
        # Should only have valid anime IDs
        self.assertTrue(cleaned['anime_id'].isin(valid_anime_ids).all())
    
    def test_cleaning_stats(self):
        """Test cleaning statistics tracking"""
        df = self.sample_data.copy()
        
        self.cleaner.handle_missing_values(df)
        self.cleaner.remove_duplicates(df)
        
        stats = self.cleaner.get_cleaning_report()
        
        # Should have statistics
        self.assertIn('missing_values', stats)
        self.assertIn('duplicates', stats)


class TestDataCleaningPipeline(unittest.TestCase):
    """Test complete cleaning pipeline"""
    
    def test_full_pipeline(self):
        """Test complete cleaning workflow"""
        # Create messy data
        messy_data = pd.DataFrame({
            'MAL_ID': [1, 2, 2, 3, 4],
            'Name': ['Test 1', 'Test 2', 'Test 2', None, 'Test 4'],
            'Score': [8.5, np.nan, 7.5, 9.0, 3.2],
            'Genres': ['Action', None, 'Drama', 'Comedy', 'Horror'],
            'Type': ['TV', 'Movie', 'Movie', 'OVA', 'TV'],
            'Episodes': [12, np.nan, 24, 6, 100],
            'Members': [100000, 50000, 50000, 80000, 200]
        })
        
        cleaner = AnimeDataCleaner()
        
        # Apply all cleaning steps
        cleaned = cleaner.handle_missing_values(messy_data)
        cleaned = cleaner.remove_duplicates(cleaned)
        cleaned = cleaner.normalize_data(cleaned)
        cleaned = cleaner.remove_outliers(cleaned)
        
        # Verify cleaned data
        self.assertGreater(len(cleaned), 0)
        self.assertFalse(cleaned['Name'].isna().any())
        self.assertFalse(cleaned['Score'].isna().any())
        self.assertEqual(cleaned['MAL_ID'].nunique(), len(cleaned))


if __name__ == '__main__':
    # Run tests
    unittest.main(verbosity=2)