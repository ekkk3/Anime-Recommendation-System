import pandas as pd
import numpy as np
from surprise import Dataset, Reader, SVD, NMF
from surprise.model_selection import train_test_split, cross_validate
from collections import defaultdict
import joblib
import logging
# Import class ModelEvaluator
from src.evaluator import ModelEvaluator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CollaborativeFilteringRecommender:
    """Collaborative Filtering using Surprise library (SVD/NMF)"""
    
    def __init__(self, algorithm='SVD', n_factors=100, n_epochs=20, 
                 lr_all=0.005, reg_all=0.02):
        """
        Initialize collaborative filtering model
        """
        self.algorithm_name = algorithm
        
        if algorithm == 'SVD':
            self.model = SVD(
                n_factors=n_factors,
                n_epochs=n_epochs,
                lr_all=lr_all,
                reg_all=reg_all,
                random_state=42
            )
        elif algorithm == 'NMF':
            self.model = NMF(
                n_factors=n_factors,
                n_epochs=n_epochs,
                random_state=42
            )
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")
        
        self.trainset = None
        self.testset = None
        self.anime_df = None
    
    def prepare_data(self, ratings_df: pd.DataFrame, anime_df: pd.DataFrame, 
                    test_size=0.2):
        """Prepare data for training"""
        logger.info("Preparing data for collaborative filtering...")
        
        self.anime_df = anime_df
        
        # Remove ratings = 0 (watching status, not actual rating)
        ratings_df = ratings_df[ratings_df['rating'] > 0].copy()
        
        # Create Surprise dataset
        reader = Reader(rating_scale=(1, 10))
        data = Dataset.load_from_df(
            ratings_df[['user_id', 'anime_id', 'rating']], 
            reader
        )
        
        # Split data
        self.trainset, self.testset = train_test_split(
            data, test_size=test_size, random_state=42
        )
        
        logger.info(f"Training set size: {self.trainset.n_ratings}")
        logger.info(f"Test set size: {len(self.testset)}")
        
        return self.trainset, self.testset
    
    def fit(self, trainset=None):
        """Train the model"""
        if trainset is not None:
            self.trainset = trainset
        
        if self.trainset is None:
            raise ValueError("No training data available. Call prepare_data first.")
        
        logger.info(f"Training {self.algorithm_name} model...")
        self.model.fit(self.trainset)
        logger.info("Model trained successfully")
    
    def predict(self, user_id: int, anime_id: int) -> float:
        """Predict rating for user-anime pair"""
        prediction = self.model.predict(user_id, anime_id)
        return prediction.est
    
    def recommend_for_user(self, user_id: int, top_n: int = 10, 
                          min_score: float = 6.0) -> pd.DataFrame:
        """Get top-N recommendations for a user"""
        if self.anime_df is None:
            raise ValueError("Anime data not available")
        
        # Get all anime IDs
        all_anime_ids = self.anime_df['MAL_ID'].values
        
        # Get anime the user has already rated
        try:
            user_rated = [
                iid for (uid, iid, _) in self.trainset.all_ratings() 
                if uid == user_id
            ]
        except:
            user_rated = []
        
        # Get unrated anime
        unrated_anime = [aid for aid in all_anime_ids if aid not in user_rated]
        
        # Filter by minimum score
        if min_score > 0:
            anime_above_threshold = self.anime_df[
                self.anime_df['Score'] >= min_score
            ]['MAL_ID'].values
            unrated_anime = [aid for aid in unrated_anime if aid in anime_above_threshold]
        
        # Predict ratings for all unrated anime
        predictions = []
        for anime_id in unrated_anime:
            pred = self.model.predict(user_id, anime_id)
            predictions.append((anime_id, pred.est))
        
        # Sort by predicted rating
        predictions.sort(key=lambda x: x[1], reverse=True)
        
        # Get top N
        top_predictions = predictions[:top_n]
        
        # Create result dataframe
        top_anime_ids = [aid for aid, _ in top_predictions]
        
        result = self.anime_df[self.anime_df['MAL_ID'].isin(top_anime_ids)].copy()
        result['predicted_rating'] = result['MAL_ID'].map(dict(top_predictions))
        result = result.sort_values('predicted_rating', ascending=False)
        
        return result[['MAL_ID', 'Name', 'Genres', 'Score', 'Type', 'Episodes', 'predicted_rating']]

    def evaluate(self, testset=None):
        """Evaluate model on test set"""
        if testset is None:
            testset = self.testset

        if testset is None:
            raise ValueError("No test data available")

        logger.info("Evaluating model...")

        # Make predictions
        predictions = self.model.test(testset)

        # Chuẩn bị dữ liệu cho ModelEvaluator
        true_ratings = np.array([pred.r_ui for pred in predictions])
        pred_ratings = np.array([pred.est for pred in predictions])

        # --- SỬ DỤNG CLASS ModelEvaluator ---
        evaluator = ModelEvaluator()

        # Tính RMSE, MAE bằng hàm có sẵn trong evaluator
        rmse = evaluator.calculate_rmse(true_ratings, pred_ratings)
        mae = evaluator.calculate_mae(true_ratings, pred_ratings)

        # --- TÍNH PRECISION/RECALL ---
        from collections import defaultdict

        # Gom nhóm gợi ý theo user
        top_n = defaultdict(list)
        for uid, iid, true_r, est, _ in predictions:
            top_n[uid].append((iid, est, true_r))

        all_recommended = []
        all_relevant = []
        threshold = 7.0 # Ngưỡng thích

        for uid, user_ratings in top_n.items():
            # Sắp xếp theo điểm dự đoán giảm dần
            user_ratings.sort(key=lambda x: x[1], reverse=True)

            # Lấy top 10 item được gợi ý
            rec_items = [iid for (iid, est, _) in user_ratings[:10]]

            # Lấy danh sách item user thực sự thích
            rel_items = [iid for (iid, _, true_r) in user_ratings if true_r >= threshold]

            all_recommended.append(rec_items)
            all_relevant.append(rel_items)

        # Gọi hàm tính Precision và Recall từ evaluator
        precisions = []
        recalls = []

        for rec, rel in zip(all_recommended, all_relevant):
            precisions.append(evaluator.precision_at_k(rec, rel, k=10))
            recalls.append(evaluator.recall_at_k(rec, rel, k=10))

        precision_10 = np.mean(precisions)
        recall_10 = np.mean(recalls)

        metrics = {
            'RMSE': rmse,
            'MAE': mae,
            'Precision@10': precision_10,
            'Recall@10': recall_10
        }

        logger.info(f"Evaluation metrics: {metrics}")
        return metrics
    
    def cross_validate(self, data, cv=5):
        """Perform cross-validation"""
        logger.info(f"Performing {cv}-fold cross-validation...")
        results = cross_validate(
            self.model, data, 
            measures=['RMSE', 'MAE'], 
            cv=cv, 
            verbose=True
        )
        return results
    
    def save_model(self, model_path: str):
        """Save the trained model"""
        model_data = {
            'model': self.model,
            'algorithm_name': self.algorithm_name,
            'trainset': self.trainset,
            'anime_df': self.anime_df
        }
        joblib.dump(model_data, model_path)
        logger.info(f"Model saved to {model_path}")
    
    def load_model(self, model_path: str):
        """Load a trained model"""
        model_data = joblib.load(model_path)
        self.model = model_data['model']
        self.algorithm_name = model_data['algorithm_name']
        self.trainset = model_data['trainset']
        self.anime_df = model_data['anime_df']
        logger.info(f"Model loaded from {model_path}")