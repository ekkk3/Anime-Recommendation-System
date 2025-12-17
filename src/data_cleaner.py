import pandas as pd
import numpy as np
from typing import Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AnimeDataCleaner:
    """Data cleaning pipeline for anime dataset"""
    
    def __init__(self):
        self.cleaned_stats = {}
    
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in anime data"""
        logger.info("Handling missing values...")
        
        initial_rows = len(df)
        
        # Fill numeric columns with median
        # Code MỚI (Đã sửa)
        numeric_cols = ['Score', 'Episodes', 'Members', 'Favorites', 'scored_by'] # Thêm scored_by cho chắc
        for col in numeric_cols:
            if col in df.columns:
                # Bước quan trọng: Chuyển chữ 'Unknown' thành NaN để không bị lỗi
                df[col] = pd.to_numeric(df[col], errors='coerce')
                
                # Tính trung vị và điền vào ô trống
                median_val = df[col].median()
                df[col] = df[col].fillna(median_val)
        
        # Fill categorical columns
        if 'Genres' in df.columns:
            df['Genres'].fillna('Unknown', inplace=True)
        
        if 'Type' in df.columns:
            df['Type'].fillna('Unknown', inplace=True)
        
        # Drop rows with missing critical fields
        df.dropna(subset=['MAL_ID', 'Name'], inplace=True)
        
        self.cleaned_stats['missing_values'] = {
            'rows_before': initial_rows,
            'rows_after': len(df),
            'rows_removed': initial_rows - len(df)
        }
        
        logger.info(f"Missing values handled. Removed {initial_rows - len(df)} rows")
        return df
    
    def remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicate entries"""
        logger.info("Removing duplicates...")
        
        initial_rows = len(df)
        
        # Remove duplicates based on MAL_ID
        df.drop_duplicates(subset=['MAL_ID'], keep='first', inplace=True)
        
        # Remove duplicates based on Name (case-insensitive)
        df['Name_lower'] = df['Name'].str.lower()
        df.drop_duplicates(subset=['Name_lower'], keep='first', inplace=True)
        df.drop(columns=['Name_lower'], inplace=True)
        
        self.cleaned_stats['duplicates'] = {
            'rows_before': initial_rows,
            'rows_after': len(df),
            'duplicates_removed': initial_rows - len(df)
        }
        
        logger.info(f"Duplicates removed. Removed {initial_rows - len(df)} rows")
        return df
    
    def normalize_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize and standardize data"""
        logger.info("Normalizing data...")
        
        # Normalize Score to 0-10 scale (if not already)
        if 'Score' in df.columns:
            df['Score'] = df['Score'].clip(0, 10)
        
        # Standardize text fields
        text_cols = ['Name', 'English name', 'Genres', 'Type']
        for col in text_cols:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip()
        
        # Convert Episodes to numeric
        if 'Episodes' in df.columns:
            df['Episodes'] = pd.to_numeric(df['Episodes'], errors='coerce')
            df['Episodes'].fillna(0, inplace=True)
        
        # Parse Genres into list
        if 'Genres' in df.columns:
            df['Genres_List'] = df['Genres'].apply(
                lambda x: [g.strip() for g in str(x).split(',')] if pd.notna(x) else []
            )
        
        logger.info("Data normalized successfully")
        return df
    
    def remove_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove outliers using IQR method"""
        logger.info("Removing outliers...")
        
        initial_rows = len(df)
        
        # Remove outliers in Members (popularity)
        if 'Members' in df.columns:
            Q1 = df['Members'].quantile(0.25)
            Q3 = df['Members'].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 3 * IQR
            upper_bound = Q3 + 3 * IQR
            df = df[(df['Members'] >= lower_bound) & (df['Members'] <= upper_bound)]
        
        # Remove anime with extremely low scores (likely invalid)
        if 'Score' in df.columns:
            df = df[df['Score'] > 0]
        
        self.cleaned_stats['outliers'] = {
            'rows_before': initial_rows,
            'rows_after': len(df),
            'outliers_removed': initial_rows - len(df)
        }
        
        logger.info(f"Outliers removed. Removed {initial_rows - len(df)} rows")
        return df
    
    def clean_ratings(self, ratings_df: pd.DataFrame, anime_ids: set) -> pd.DataFrame:
        """Clean rating data"""
        logger.info("Cleaning ratings data...")
        
        initial_rows = len(ratings_df)
        
        # Remove ratings for non-existent anime
        ratings_df = ratings_df[ratings_df['anime_id'].isin(anime_ids)]
        
        # Remove invalid ratings (not in 1-10 range, excluding 0 for watching)
        ratings_df = ratings_df[
            ((ratings_df['rating'] >= 1) & (ratings_df['rating'] <= 10)) | 
            (ratings_df['rating'] == 0)
        ]
        
        # Remove duplicates
        ratings_df.drop_duplicates(subset=['user_id', 'anime_id'], keep='last', inplace=True)
        
        logger.info(f"Ratings cleaned. Removed {initial_rows - len(ratings_df)} rows")
        return ratings_df
    
    def get_cleaning_report(self) -> dict:
        """Get summary of cleaning operations"""
        return self.cleaned_stats


def clean_anime_data(anime_path: str, output_path: str) -> pd.DataFrame:
    """Main function to clean anime data"""
    logger.info(f"Loading anime data from {anime_path}")
    df = pd.read_csv(anime_path)
    
    cleaner = AnimeDataCleaner()
    
    # Apply cleaning steps
    df = cleaner.handle_missing_values(df)
    df = cleaner.remove_duplicates(df)
    df = cleaner.normalize_data(df)
    df = cleaner.remove_outliers(df)
    
    # Save cleaned data
    df.to_csv(output_path, index=False)
    logger.info(f"Cleaned data saved to {output_path}")
    
    # Print cleaning report
    report = cleaner.get_cleaning_report()
    logger.info(f"Cleaning Report: {report}")
    
    return df


if __name__ == "__main__":
    from config import ANIME_RAW, ANIME_CLEANED
    
    cleaned_df = clean_anime_data(ANIME_RAW, ANIME_CLEANED)
    print(f"\nFinal dataset shape: {cleaned_df.shape}")
    print(f"\nData info:")
    print(cleaned_df.info())