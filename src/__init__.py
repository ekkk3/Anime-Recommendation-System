"""
Anime Recommendation System - Source Package
"""

__version__ = "1.0.0"
__author__ = "Your Name"

from .data_loader import load_anime_data, load_ratings_data, sample_ratings
from .data_cleaner import AnimeDataCleaner, clean_anime_data
from .feature_engineer import FeatureEngineer
from .evaluator import ModelEvaluator
from .utils import *

__all__ = [
    'load_anime_data',
    'load_ratings_data',
    'sample_ratings',
    'AnimeDataCleaner',
    'clean_anime_data',
    'FeatureEngineer',
    'ModelEvaluator'
]