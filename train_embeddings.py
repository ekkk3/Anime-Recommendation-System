"""
Script to train advanced embeddings
Run: python train_embeddings.py
"""

import sys
from pathlib import Path
import pandas as pd
import logging

sys.path.append(str(Path(__file__).parent))

from src.advanced_embeddings import AdvancedEmbeddings
from config import ANIME_CLEANED, MODEL_DIR, VIZ_DIR

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Train and save advanced embeddings"""
    
    logger.info("=" * 70)
    logger.info("TRAINING ADVANCED EMBEDDINGS")
    logger.info("=" * 70)
    
    # Step 1: Load data
    logger.info("\n[STEP 1/5] Loading anime data...")
    try:
        anime_df = pd.read_csv(ANIME_CLEANED)
        logger.info(f"✅ Loaded {len(anime_df):,} anime")
    except Exception as e:
        logger.error(f"❌ Failed to load data: {e}")
        return
    
    # Step 2: Initialize embedder
    logger.info("\n[STEP 2/5] Initializing embedding model...")
    embedder = AdvancedEmbeddings(
        vector_size=100,  # Dimension of embeddings
        window=5,         # Context window
        min_count=2       # Minimum frequency
    )
    logger.info("✅ Model initialized")
    
    # Step 3: Train embeddings
    logger.info("\n[STEP 3/5] Training Word2Vec on anime genres...")
    try:
        embeddings = embedder.train_genre_embeddings(anime_df)
        logger.info(f"✅ Generated embeddings for {len(embeddings):,} anime")
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        return
    
    # Step 4: Save model
    logger.info("\n[STEP 4/5] Saving trained model...")
    model_path = MODEL_DIR / "w2v_embeddings.model"
    try:
        embedder.save_embeddings(str(model_path))
        logger.info(f"✅ Model saved to {model_path}")
    except Exception as e:
        logger.error(f"❌ Save failed: {e}")
    
    # Step 5: Create visualization
    logger.info("\n[STEP 5/5] Creating t-SNE visualization...")
    viz_path = VIZ_DIR / "genre_embeddings_tsne.png"
    try:
        embedder.visualize_genre_space(str(viz_path))
        logger.info(f"✅ Visualization saved to {viz_path}")
    except Exception as e:
        logger.warning(f"⚠️ Visualization failed: {e}")
    
    # Test recommendations
    logger.info("\n" + "=" * 70)
    logger.info("TESTING EMBEDDINGS")
    logger.info("=" * 70)
    
    # Pick a random anime
    test_anime = anime_df.sample(1).iloc[0]
    test_id = test_anime['MAL_ID']
    test_name = test_anime['Name']
    test_genres = test_anime['Genres']
    
    logger.info(f"\n🎬 Test Anime: {test_name}")
    logger.info(f"   ID: {test_id}")
    logger.info(f"   Genres: {test_genres}")
    
    # Get similar anime
    try:
        similar = embedder.get_similar_by_embedding(test_id, top_n=10)
        
        logger.info(f"\n📊 Top 10 Similar Anime (by embeddings):")
        logger.info("-" * 70)
        
        for rank, (mal_id, score) in enumerate(similar, 1):
            anime_info = anime_df[anime_df['MAL_ID'] == mal_id].iloc[0]
            logger.info(f"{rank:2d}. {anime_info['Name']:<40} (Similarity: {score:.4f})")
            logger.info(f"    Genres: {anime_info['Genres']}")
            logger.info(f"    Score: {anime_info['Score']:.2f}/10")
            logger.info("")
        
    except Exception as e:
        logger.error(f"❌ Testing failed: {e}")
    
    # Summary
    logger.info("=" * 70)
    logger.info("✅ ADVANCED EMBEDDINGS TRAINING COMPLETED!")
    logger.info("=" * 70)
    logger.info("\n📁 Generated Files:")
    logger.info(f"   - Model: {model_path}")
    logger.info(f"   - Visualization: {viz_path}")
    logger.info("\n🎯 Next Steps:")
    logger.info("   1. Review visualization in data/visualizations/")
    logger.info("   2. Add embeddings to recommendation pipeline")
    logger.info("   3. Compare with TF-IDF baseline")
    logger.info("   4. Document in report for bonus points!")
    logger.info("")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("\n\n⚠️ Training interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"\n\n❌ Unexpected error: {e}", exc_info=True)
        sys.exit(1)