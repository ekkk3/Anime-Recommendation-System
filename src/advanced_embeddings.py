"""
Advanced Embeddings for Anime Recommendation
Using Word2Vec and Sentence Transformers
"""

import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from sklearn.preprocessing import normalize
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AdvancedEmbeddings:
    """
    Advanced embedding techniques for anime features
    """
    
    def __init__(self, vector_size=100, window=5, min_count=2):
        """
        Initialize embedding model
        
        Args:
            vector_size: Dimension of embedding vectors
            window: Context window size
            min_count: Minimum word frequency
        """
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.w2v_model = None
        self.genre_embeddings = None
        
    def train_genre_embeddings(self, anime_df):
        """
        Train Word2Vec embeddings on anime genres
        
        Args:
            anime_df: Anime dataframe with 'Genres' column
            
        Returns:
            Dictionary mapping anime_id to embedding vector
        """
        logger.info("Training Word2Vec embeddings on genres...")
        
        # Prepare genre sequences
        genre_sequences = []
        for genres in anime_df['Genres'].dropna():
            genre_list = [g.strip().lower().replace(' ', '_') 
                         for g in str(genres).split(',')]
            genre_sequences.append(genre_list)
        
        # Train Word2Vec
        self.w2v_model = Word2Vec(
            sentences=genre_sequences,
            vector_size=self.vector_size,
            window=self.window,
            min_count=self.min_count,
            workers=4,
            epochs=20,
            sg=1  # Skip-gram
        )
        
        logger.info(f"Word2Vec trained. Vocabulary size: {len(self.w2v_model.wv)}")
        
        # Generate anime embeddings
        anime_embeddings = {}
        
        for idx, row in anime_df.iterrows():
            mal_id = row['MAL_ID']
            genres = row['Genres']
            
            if pd.notna(genres):
                genre_list = [g.strip().lower().replace(' ', '_') 
                            for g in str(genres).split(',')]
                
                # Average word vectors for this anime
                vectors = []
                for genre in genre_list:
                    if genre in self.w2v_model.wv:
                        vectors.append(self.w2v_model.wv[genre])
                
                if vectors:
                    anime_embeddings[mal_id] = np.mean(vectors, axis=0)
                else:
                    # Fallback to zero vector
                    anime_embeddings[mal_id] = np.zeros(self.vector_size)
            else:
                anime_embeddings[mal_id] = np.zeros(self.vector_size)
        
        self.genre_embeddings = anime_embeddings
        logger.info(f"Generated embeddings for {len(anime_embeddings)} anime")
        
        return anime_embeddings
    
    def get_similar_by_embedding(self, anime_id, top_n=10):
        """
        Find similar anime using embedding similarity
        
        Args:
            anime_id: Target anime ID
            top_n: Number of similar anime to return
            
        Returns:
            List of (anime_id, similarity_score) tuples
        """
        if self.genre_embeddings is None:
            raise ValueError("Embeddings not trained. Call train_genre_embeddings first.")
        
        if anime_id not in self.genre_embeddings:
            logger.warning(f"Anime {anime_id} not found in embeddings")
            return []
        
        target_embedding = self.genre_embeddings[anime_id]
        
        # Calculate cosine similarities
        similarities = []
        for mal_id, embedding in self.genre_embeddings.items():
            if mal_id != anime_id:
                # Cosine similarity
                sim = np.dot(target_embedding, embedding) / (
                    np.linalg.norm(target_embedding) * np.linalg.norm(embedding) + 1e-10
                )
                similarities.append((mal_id, sim))
        
        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities[:top_n]
    
    def visualize_genre_space(self, output_path='genre_embeddings_2d.png'):
        """
        Visualize genre embeddings using t-SNE
        
        Args:
            output_path: Path to save visualization
        """
        if self.w2v_model is None:
            logger.warning("Model not trained")
            return
        
        try:
            from sklearn.manifold import TSNE
            import matplotlib.pyplot as plt
            
            # Get genre vectors
            genres = list(self.w2v_model.wv.key_to_index.keys())
            vectors = np.array([self.w2v_model.wv[g] for g in genres])
            
            # Reduce to 2D using t-SNE
            logger.info("Running t-SNE dimensionality reduction...")
            tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(genres)-1))
            vectors_2d = tsne.fit_transform(vectors)
            
            # Plot
            plt.figure(figsize=(12, 10))
            plt.scatter(vectors_2d[:, 0], vectors_2d[:, 1], alpha=0.6)
            
            # Annotate genres
            for i, genre in enumerate(genres):
                plt.annotate(
                    genre.replace('_', ' ').title(),
                    xy=(vectors_2d[i, 0], vectors_2d[i, 1]),
                    fontsize=8,
                    alpha=0.7
                )
            
            plt.title('Anime Genre Embeddings (t-SNE 2D)', fontsize=16, fontweight='bold')
            plt.xlabel('Dimension 1')
            plt.ylabel('Dimension 2')
            plt.grid(alpha=0.3)
            plt.tight_layout()
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Visualization saved to {output_path}")
            
        except ImportError:
            logger.warning("sklearn or matplotlib not available for visualization")
    
    def save_embeddings(self, model_path='models/w2v_genre_embeddings.model'):
        """Save trained Word2Vec model"""
        if self.w2v_model:
            self.w2v_model.save(model_path)
            logger.info(f"Embeddings saved to {model_path}")
    
    def load_embeddings(self, model_path='models/w2v_genre_embeddings.model'):
        """Load pre-trained Word2Vec model"""
        self.w2v_model = Word2Vec.load(model_path)
        logger.info(f"Embeddings loaded from {model_path}")


def demonstrate_advanced_embeddings():
    """
    Demonstration of advanced embeddings
    """
    from config import ANIME_CLEANED
    
    # Load data
    anime_df = pd.read_csv(ANIME_CLEANED)
    
    # Initialize and train embeddings
    embedder = AdvancedEmbeddings(vector_size=100, window=5, min_count=2)
    
    # Train on genres
    embeddings = embedder.train_genre_embeddings(anime_df)
    
    # Test similarity
    test_anime_id = anime_df.iloc[0]['MAL_ID']
    test_anime_name = anime_df.iloc[0]['Name']
    
    logger.info(f"\nTesting similarity for: {test_anime_name} (ID: {test_anime_id})")
    
    similar = embedder.get_similar_by_embedding(test_anime_id, top_n=10)
    
    logger.info("\nTop 10 similar anime (by embeddings):")
    for mal_id, score in similar:
        anime_info = anime_df[anime_df['MAL_ID'] == mal_id].iloc[0]
        logger.info(f"  - {anime_info['Name']} (Score: {score:.3f})")
    
    # Visualize
    embedder.visualize_genre_space('data/visualizations/genre_embeddings.png')
    
    # Save model
    embedder.save_embeddings('models/w2v_embeddings.model')
    
    logger.info("\n✅ Advanced embeddings demonstration completed!")


if __name__ == "__main__":
    demonstrate_advanced_embeddings()