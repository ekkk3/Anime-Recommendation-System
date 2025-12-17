"""
Training script for Anime Recommendation System
Trains Content-Based, Collaborative Filtering, and Hybrid models
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent))

from src.data_cleaner import clean_anime_data
from src.models.content_based import ContentBasedRecommender
from src.models.collaborative import CollaborativeFilteringRecommender
from src.models.hybrid_recommender import HybridRecommender
from src.utils import create_all_visualizations
from config import *

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def sample_ratings_data(rating_path, output_path, sample_size=5_000_000):
    """Sample large rating file for training"""
    logger.info(f"Sampling {sample_size:,} ratings from {rating_path}")
    
    chunk_size = 1_000_000
    chunks = []
    rows_read = 0
    
    for chunk in pd.read_csv(rating_path, chunksize=chunk_size):
        # Remove watching status (rating = 0)
        chunk = chunk[chunk['rating'] > 0]
        chunks.append(chunk)
        rows_read += len(chunk)
        
        logger.info(f"Read {rows_read:,} rows...")
        
        if rows_read >= sample_size:
            break
    
    # Combine chunks
    ratings_df = pd.concat(chunks, ignore_index=True)
    
    # Sample if we have more than needed
    if len(ratings_df) > sample_size:
        ratings_df = ratings_df.sample(n=sample_size, random_state=RANDOM_STATE)
    
    # Save
    ratings_df.to_csv(output_path, index=False)
    logger.info(f"Sampled {len(ratings_df):,} ratings saved to {output_path}")
    
    return ratings_df


def filter_popular_anime(anime_df, ratings_df, min_ratings=MIN_RATING_COUNT):
    """Filter anime with minimum ratings"""
    logger.info(f"Filtering anime with at least {min_ratings} ratings...")
    
    anime_rating_counts = ratings_df['anime_id'].value_counts()
    popular_anime_ids = anime_rating_counts[anime_rating_counts >= min_ratings].index
    
    filtered_anime_df = anime_df[anime_df['MAL_ID'].isin(popular_anime_ids)].copy()
    filtered_ratings_df = ratings_df[ratings_df['anime_id'].isin(popular_anime_ids)].copy()
    
    logger.info(f"Anime: {len(anime_df)} -> {len(filtered_anime_df)}")
    logger.info(f"Ratings: {len(ratings_df)} -> {len(filtered_ratings_df)}")
    
    return filtered_anime_df, filtered_ratings_df


def train_content_based_model(anime_df):
    """Train content-based model"""
    logger.info("=" * 70)
    logger.info("TRAINING CONTENT-BASED MODEL")
    logger.info("=" * 70)
    
    model = ContentBasedRecommender(
        max_features=CONTENT_BASED_CONFIG['tfidf_max_features'],
        ngram_range=CONTENT_BASED_CONFIG['tfidf_ngram_range']
    )
    
    model.fit(anime_df)
    model.save_model(CONTENT_MODEL, TFIDF_VECTORIZER)
    
    # Test
    logger.info("\n--- Testing Content-Based Model ---")
    test_anime_id = anime_df.iloc[0]['MAL_ID']
    test_anime_name = anime_df.iloc[0]['Name']
    logger.info(f"Test anime: {test_anime_name} (ID: {test_anime_id})")
    
    recommendations = model.recommend(test_anime_id, top_n=5)
    logger.info(f"\nTop 5 recommendations:")
    for idx, row in recommendations.iterrows():
        logger.info(f"  - {row['Name']} (Score: {row['Score']:.2f}, Sim: {row['similarity_score']:.3f})")
    
    return model


def train_collaborative_model(anime_df, ratings_df):
    """Train collaborative filtering model"""
    logger.info("=" * 70)
    logger.info("TRAINING COLLABORATIVE FILTERING MODEL")
    logger.info("=" * 70)
    
    model = CollaborativeFilteringRecommender(
        algorithm=COLLABORATIVE_CONFIG['algorithm'],
        n_factors=COLLABORATIVE_CONFIG['n_factors'],
        n_epochs=COLLABORATIVE_CONFIG['n_epochs'],
        lr_all=COLLABORATIVE_CONFIG['lr_all'],
        reg_all=COLLABORATIVE_CONFIG['reg_all']
    )
    
    trainset, testset = model.prepare_data(ratings_df, anime_df, test_size=1-TRAIN_TEST_SPLIT)
    model.fit(trainset)
    
    # Evaluate
    logger.info("\n--- Evaluating Collaborative Model ---")
    metrics = model.evaluate(testset)
    logger.info(f"Test RMSE: {metrics['RMSE']:.4f}")
    logger.info(f"Test MAE: {metrics['MAE']:.4f}")
    
    # --- DÁN THÊM 2 DÒNG NÀY ---
    logger.info(f"Test Precision@10: {metrics['Precision@10']:.4f}")
    logger.info(f"Test Recall@10: {metrics['Recall@10']:.4f}")
    # ---------------------------

    model.save_model(COLLABORATIVE_MODEL)
    
    # Test
    logger.info("\n--- Testing Collaborative Model ---")
    test_user_id = ratings_df.iloc[0]['user_id']
    logger.info(f"Test user ID: {test_user_id}")
    
    try:
        recommendations = model.recommend_for_user(test_user_id, top_n=5)
        logger.info(f"\nTop 5 recommendations:")
        for idx, row in recommendations.iterrows():
            logger.info(f"  - {row['Name']} (Score: {row['Score']:.2f}, Predicted: {row['predicted_rating']:.2f})")
    except Exception as e:
        logger.warning(f"Could not get recommendations: {str(e)}")
    
    return model


def train_hybrid_model(content_model, collaborative_model):
    """Train hybrid model"""
    logger.info("=" * 70)
    logger.info("TRAINING HYBRID MODEL")
    logger.info("=" * 70)
    
    hybrid_model = HybridRecommender(
        content_model=content_model,
        collaborative_model=collaborative_model,
        content_weight=HYBRID_CONFIG['content_weight'],
        collaborative_weight=HYBRID_CONFIG['collaborative_weight'],
        strategy='weighted'
    )
    
    hybrid_model.save_model(HYBRID_MODEL)
    
    # Test
    logger.info("\n--- Testing Hybrid Model ---")
    test_anime_id = content_model.anime_df.iloc[0]['MAL_ID']
    test_anime_name = content_model.anime_df.iloc[0]['Name']
    logger.info(f"Test anime: {test_anime_name} (ID: {test_anime_id})")
    
    recommendations = hybrid_model.recommend(anime_id=test_anime_id, top_n=5)
    logger.info(f"\nTop 5 hybrid recommendations:")
    for idx, row in recommendations.iterrows():
        logger.info(f"  - {row['Name']}")
        logger.info(f"    Score: {row['Score']:.2f} | CB: {row['cb_score']:.3f} | CF: {row['cf_score']:.3f} | Hybrid: {row['hybrid_score']:.3f}")
    
    return hybrid_model


def create_visualizations(anime_df):
    """Create all visualizations"""
    logger.info("=" * 70)
    logger.info("CREATING VISUALIZATIONS")
    logger.info("=" * 70)
    
    try:
        create_all_visualizations(anime_df, output_dir=VIZ_DIR)
        logger.info(f"✅ Visualizations saved to {VIZ_DIR}")
    except Exception as e:
        logger.error(f"❌ Error creating visualizations: {str(e)}")


def main():
    """Main training pipeline"""
    logger.info("\n" + "=" * 70)
    logger.info("ANIME RECOMMENDATION SYSTEM - TRAINING PIPELINE")
    logger.info("=" * 70 + "\n")
    
    # Step 1: Clean anime data
    logger.info("[STEP 1/6] Cleaning anime data...")
    if not ANIME_CLEANED.exists():
        anime_df = clean_anime_data(ANIME_RAW, ANIME_CLEANED)
    else:
        logger.info(f"✅ Cleaned data exists at {ANIME_CLEANED}")
        anime_df = pd.read_csv(ANIME_CLEANED)
    
    logger.info(f"📊 Anime dataset: {anime_df.shape}")
    
    # Step 2: Sample rating data
    logger.info("\n[STEP 2/6] Sampling rating data...")
    if not RATING_SAMPLED.exists():
        ratings_df = sample_ratings_data(RATING_RAW, RATING_SAMPLED, RATING_SAMPLE_SIZE)
    else:
        logger.info(f"✅ Sampled ratings exist at {RATING_SAMPLED}")
        ratings_df = pd.read_csv(RATING_SAMPLED)
    
    logger.info(f"📊 Ratings dataset: {ratings_df.shape}")
    
    # Step 3: Filter popular anime
    logger.info("\n[STEP 3/6] Filtering popular anime...")
    anime_df, ratings_df = filter_popular_anime(anime_df, ratings_df)
    
    # Save filtered data
    anime_df.to_csv(ANIME_CLEANED, index=False)
    ratings_df.to_csv(RATING_SAMPLED, index=False)
    
    # Step 4: Train content-based model
    logger.info("\n[STEP 4/6] Training Content-Based model...")
    content_model = train_content_based_model(anime_df)
    
    # Step 5: Train collaborative model
    logger.info("\n[STEP 5/6] Training Collaborative Filtering model...")
    cf_model = train_collaborative_model(anime_df, ratings_df)
    
    # Step 6: Train hybrid model
    logger.info("\n[STEP 6/6] Training Hybrid model...")
    hybrid_model = train_hybrid_model(content_model, cf_model)
    
    # Bonus: Create visualizations
    logger.info("\n[BONUS] Creating visualizations...")
    create_visualizations(anime_df)
    
    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("✅ TRAINING COMPLETED SUCCESSFULLY!")
    logger.info("=" * 70)
    
    logger.info("\n📦 Trained Models:")
    logger.info(f"  ✅ Content-Based: {CONTENT_MODEL}")
    logger.info(f"  ✅ Collaborative: {COLLABORATIVE_MODEL}")
    logger.info(f"  ✅ Hybrid: {HYBRID_MODEL}")
    logger.info(f"  ✅ TF-IDF Vectorizer: {TFIDF_VECTORIZER}")
    
    logger.info("\n📊 Processed Data:")
    logger.info(f"  ✅ Cleaned anime: {ANIME_CLEANED}")
    logger.info(f"  ✅ Sampled ratings: {RATING_SAMPLED}")
    
    logger.info("\n📈 Visualizations:")
    logger.info(f"  ✅ Saved to: {VIZ_DIR}")
    
    logger.info("\n🚀 Next Steps:")
    logger.info("  1. Review visualizations in data/visualizations/")
    logger.info("  2. Run the web app:")
    logger.info("     $ python run_app.py")
    logger.info("     or")
    logger.info("     $ streamlit run app/main.py")
    logger.info("  3. Login as admin (username: admin, password: admin123)")
    logger.info("  4. Switch between models and test recommendations!")
    
    logger.info("\n" + "=" * 70 + "\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("\n\n⚠️ Training interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"\n\n❌ Training failed: {str(e)}", exc_info=True)
        sys.exit(1)