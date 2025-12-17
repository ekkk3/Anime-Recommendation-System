import os
from pathlib import Path

# Base Directory
BASE_DIR = Path(__file__).parent

# Data Directories
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
VIZ_DIR = DATA_DIR / "visualizations"

# Model Directory
MODEL_DIR = BASE_DIR / "models"

# Create directories if not exist
for dir_path in [RAW_DATA_DIR, PROCESSED_DATA_DIR, VIZ_DIR, MODEL_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Data File Paths
ANIME_RAW = RAW_DATA_DIR / "anime.csv"
RATING_RAW = RAW_DATA_DIR / "rating_complete.csv"
ANIMELIST_RAW = RAW_DATA_DIR / "animelist.csv"

ANIME_CLEANED = PROCESSED_DATA_DIR / "anime_cleaned.csv"
RATING_SAMPLED = PROCESSED_DATA_DIR / "ratings_sampled.csv"
USER_ITEM_MATRIX = PROCESSED_DATA_DIR / "user_item_matrix.npz"
ANIME_FEATURES = PROCESSED_DATA_DIR / "anime_features.pkl"

# Model Paths
CONTENT_MODEL = MODEL_DIR / "content_model.pkl"
COLLABORATIVE_MODEL = MODEL_DIR / "collaborative_model.pkl"
HYBRID_MODEL = MODEL_DIR / "hybrid_model.pkl"
TFIDF_VECTORIZER = MODEL_DIR / "tfidf_vectorizer.pkl"
SIMILARITY_MATRIX = MODEL_DIR / "similarity_matrix.npz"

# Model Parameters
CONTENT_BASED_CONFIG = {
    "tfidf_max_features": 5000,
    "tfidf_ngram_range": (1, 2),
    "similarity_metric": "cosine"
}

COLLABORATIVE_CONFIG = {
    "algorithm": "SVD",  # SVD, NMF, or LightFM
    "n_factors": 100,
    "n_epochs": 20,
    "lr_all": 0.005,
    "reg_all": 0.02
}

HYBRID_CONFIG = {
    "content_weight": 0.4,
    "collaborative_weight": 0.6
}

# Data Processing Parameters
MIN_RATING_COUNT = 100  # Minimum ratings for anime to be included
RATING_SAMPLE_SIZE = 5_000_000  # Sample size for large rating file
TRAIN_TEST_SPLIT = 0.8

# Recommendation Parameters
TOP_N_RECOMMENDATIONS = 10
MIN_SCORE_THRESHOLD = 6.0

# Streamlit Config
APP_TITLE = "🎌 Anime Recommendation System"
APP_ICON = "🎌"
PAGE_LAYOUT = "wide"

# Random Seed
RANDOM_STATE = 42