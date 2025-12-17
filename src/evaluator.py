"""
Model evaluation metrics for recommendation systems
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
import logging
from collections import defaultdict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Comprehensive evaluation metrics for recommendation models"""
    
    def __init__(self):
        self.metrics = {}
    
    def calculate_rmse(self, true_ratings: np.ndarray, predicted_ratings: np.ndarray) -> float:
        """
        Calculate Root Mean Squared Error
        
        Args:
            true_ratings: Actual ratings
            predicted_ratings: Predicted ratings
            
        Returns:
            RMSE score
        """
        mse = np.mean((true_ratings - predicted_ratings) ** 2)
        rmse = np.sqrt(mse)
        logger.info(f"RMSE: {rmse:.4f}")
        return rmse
    
    def calculate_mae(self, true_ratings: np.ndarray, predicted_ratings: np.ndarray) -> float:
        """
        Calculate Mean Absolute Error
        
        Args:
            true_ratings: Actual ratings
            predicted_ratings: Predicted ratings
            
        Returns:
            MAE score
        """
        mae = np.mean(np.abs(true_ratings - predicted_ratings))
        logger.info(f"MAE: {mae:.4f}")
        return mae
    
    def calculate_mse(self, true_ratings: np.ndarray, predicted_ratings: np.ndarray) -> float:
        """
        Calculate Mean Squared Error
        
        Args:
            true_ratings: Actual ratings
            predicted_ratings: Predicted ratings
            
        Returns:
            MSE score
        """
        mse = np.mean((true_ratings - predicted_ratings) ** 2)
        logger.info(f"MSE: {mse:.4f}")
        return mse
    
    def precision_at_k(self, recommended_items: List, relevant_items: List, k: int) -> float:
        """
        Calculate Precision@K
        
        Precision@K = (# of recommended items @k that are relevant) / k
        
        Args:
            recommended_items: List of recommended item IDs
            relevant_items: List of relevant item IDs
            k: Number of top recommendations to consider
            
        Returns:
            Precision@K score
        """
        if k <= 0 or len(recommended_items) == 0:
            return 0.0
        
        recommended_at_k = recommended_items[:k]
        relevant_set = set(relevant_items)
        
        num_relevant = len([item for item in recommended_at_k if item in relevant_set])
        precision = num_relevant / k
        
        return precision
    
    def recall_at_k(self, recommended_items: List, relevant_items: List, k: int) -> float:
        """
        Calculate Recall@K
        
        Recall@K = (# of recommended items @k that are relevant) / (total # of relevant items)
        
        Args:
            recommended_items: List of recommended item IDs
            relevant_items: List of relevant item IDs
            k: Number of top recommendations to consider
            
        Returns:
            Recall@K score
        """
        if len(relevant_items) == 0 or len(recommended_items) == 0:
            return 0.0
        
        recommended_at_k = recommended_items[:k]
        relevant_set = set(relevant_items)
        
        num_relevant = len([item for item in recommended_at_k if item in relevant_set])
        recall = num_relevant / len(relevant_items)
        
        return recall
    
    def f1_score_at_k(self, recommended_items: List, relevant_items: List, k: int) -> float:
        """
        Calculate F1-Score@K
        
        F1@K = 2 * (Precision@K * Recall@K) / (Precision@K + Recall@K)
        
        Args:
            recommended_items: List of recommended item IDs
            relevant_items: List of relevant item IDs
            k: Number of top recommendations to consider
            
        Returns:
            F1-Score@K
        """
        precision = self.precision_at_k(recommended_items, relevant_items, k)
        recall = self.recall_at_k(recommended_items, relevant_items, k)
        
        if precision + recall == 0:
            return 0.0
        
        f1 = 2 * (precision * recall) / (precision + recall)
        return f1
    
    def average_precision_at_k(self, recommended_items: List, relevant_items: List, k: int) -> float:
        """
        Calculate Average Precision@K (AP@K)
        
        Args:
            recommended_items: List of recommended item IDs
            relevant_items: List of relevant item IDs
            k: Number of top recommendations to consider
            
        Returns:
            Average Precision@K score
        """
        if len(relevant_items) == 0:
            return 0.0
        
        recommended_at_k = recommended_items[:k]
        relevant_set = set(relevant_items)
        
        score = 0.0
        num_hits = 0.0
        
        for i, item in enumerate(recommended_at_k):
            if item in relevant_set:
                num_hits += 1.0
                score += num_hits / (i + 1.0)
        
        if num_hits == 0:
            return 0.0
        
        return score / min(len(relevant_items), k)
    
    def mean_average_precision_at_k(self, all_recommended: List[List], 
                                    all_relevant: List[List], k: int) -> float:
        """
        Calculate Mean Average Precision@K (MAP@K)
        
        Args:
            all_recommended: List of recommendation lists for each user
            all_relevant: List of relevant item lists for each user
            k: Number of top recommendations to consider
            
        Returns:
            MAP@K score
        """
        ap_scores = []
        
        for recommended, relevant in zip(all_recommended, all_relevant):
            ap = self.average_precision_at_k(recommended, relevant, k)
            ap_scores.append(ap)
        
        map_score = np.mean(ap_scores)
        logger.info(f"MAP@{k}: {map_score:.4f}")
        return map_score
    
    def ndcg_at_k(self, recommended_items: List, relevant_items: Dict[int, float], k: int) -> float:
        """
        Calculate Normalized Discounted Cumulative Gain@K (NDCG@K)
        
        Args:
            recommended_items: List of recommended item IDs
            relevant_items: Dict mapping item IDs to relevance scores
            k: Number of top recommendations to consider
            
        Returns:
            NDCG@K score
        """
        def dcg_at_k(items, relevances, k):
            """Calculate DCG@K"""
            dcg = 0.0
            for i, item in enumerate(items[:k]):
                relevance = relevances.get(item, 0)
                dcg += (2 ** relevance - 1) / np.log2(i + 2)
            return dcg
        
        # Calculate DCG
        dcg = dcg_at_k(recommended_items, relevant_items, k)
        
        # Calculate IDCG (Ideal DCG)
        ideal_items = sorted(relevant_items.keys(), 
                           key=lambda x: relevant_items[x], 
                           reverse=True)
        idcg = dcg_at_k(ideal_items, relevant_items, k)
        
        # Calculate NDCG
        if idcg == 0:
            return 0.0
        
        ndcg = dcg / idcg
        return ndcg
    
    def hit_rate_at_k(self, all_recommended: List[List], 
                     all_relevant: List[List], k: int) -> float:
        """
        Calculate Hit Rate@K
        
        Hit Rate@K = (# of users with at least 1 relevant item in top K) / (total # of users)
        
        Args:
            all_recommended: List of recommendation lists for each user
            all_relevant: List of relevant item lists for each user
            k: Number of top recommendations to consider
            
        Returns:
            Hit Rate@K score
        """
        hits = 0
        
        for recommended, relevant in zip(all_recommended, all_relevant):
            recommended_at_k = set(recommended[:k])
            relevant_set = set(relevant)
            
            if len(recommended_at_k & relevant_set) > 0:
                hits += 1
        
        hit_rate = hits / len(all_recommended) if len(all_recommended) > 0 else 0.0
        logger.info(f"Hit Rate@{k}: {hit_rate:.4f}")
        return hit_rate
    
    def coverage(self, all_recommended: List[List], total_items: int) -> float:
        """
        Calculate catalog coverage
        
        Coverage = (# of unique items recommended) / (total # of items)
        
        Args:
            all_recommended: List of recommendation lists
            total_items: Total number of items in catalog
            
        Returns:
            Coverage score
        """
        unique_items = set()
        for recommended in all_recommended:
            unique_items.update(recommended)
        
        coverage = len(unique_items) / total_items if total_items > 0 else 0.0
        logger.info(f"Coverage: {coverage:.4f}")
        return coverage
    
    def diversity(self, all_recommended: List[List], 
                 similarity_matrix: np.ndarray, 
                 item_to_idx: Dict) -> float:
        """
        Calculate diversity of recommendations
        
        Diversity = average pairwise distance between recommended items
        
        Args:
            all_recommended: List of recommendation lists
            similarity_matrix: Item-item similarity matrix
            item_to_idx: Mapping from item IDs to matrix indices
            
        Returns:
            Diversity score (higher is more diverse)
        """
        diversity_scores = []
        
        for recommended in all_recommended:
            if len(recommended) < 2:
                continue
            
            # Calculate pairwise dissimilarity
            dissimilarities = []
            for i in range(len(recommended)):
                for j in range(i + 1, len(recommended)):
                    item_i = recommended[i]
                    item_j = recommended[j]
                    
                    if item_i in item_to_idx and item_j in item_to_idx:
                        idx_i = item_to_idx[item_i]
                        idx_j = item_to_idx[item_j]
                        similarity = similarity_matrix[idx_i, idx_j]
                        dissimilarity = 1 - similarity
                        dissimilarities.append(dissimilarity)
            
            if dissimilarities:
                diversity_scores.append(np.mean(dissimilarities))
        
        avg_diversity = np.mean(diversity_scores) if diversity_scores else 0.0
        logger.info(f"Diversity: {avg_diversity:.4f}")
        return avg_diversity
    
    def evaluate_all(self, true_ratings: np.ndarray, 
                    predicted_ratings: np.ndarray,
                    all_recommended: List[List] = None,
                    all_relevant: List[List] = None,
                    k_values: List[int] = [5, 10, 20]) -> Dict:
        """
        Calculate all evaluation metrics
        
        Args:
            true_ratings: Actual ratings
            predicted_ratings: Predicted ratings
            all_recommended: List of recommendation lists (optional)
            all_relevant: List of relevant item lists (optional)
            k_values: List of K values for @K metrics
            
        Returns:
            Dictionary with all metrics
        """
        logger.info("=" * 50)
        logger.info("Evaluating Model Performance")
        logger.info("=" * 50)
        
        metrics = {}
        
        # Rating prediction metrics
        metrics['RMSE'] = self.calculate_rmse(true_ratings, predicted_ratings)
        metrics['MAE'] = self.calculate_mae(true_ratings, predicted_ratings)
        metrics['MSE'] = self.calculate_mse(true_ratings, predicted_ratings)
        
        # Ranking metrics (if provided)
        if all_recommended is not None and all_relevant is not None:
            for k in k_values:
                # Calculate precision, recall, F1 for each user
                precisions = []
                recalls = []
                f1_scores = []
                
                for rec, rel in zip(all_recommended, all_relevant):
                    precisions.append(self.precision_at_k(rec, rel, k))
                    recalls.append(self.recall_at_k(rec, rel, k))
                    f1_scores.append(self.f1_score_at_k(rec, rel, k))
                
                metrics[f'Precision@{k}'] = np.mean(precisions)
                metrics[f'Recall@{k}'] = np.mean(recalls)
                metrics[f'F1@{k}'] = np.mean(f1_scores)
                
                # MAP and Hit Rate
                metrics[f'MAP@{k}'] = self.mean_average_precision_at_k(
                    all_recommended, all_relevant, k
                )
                metrics[f'HitRate@{k}'] = self.hit_rate_at_k(
                    all_recommended, all_relevant, k
                )
        
        # Print summary
        logger.info("\n" + "=" * 50)
        logger.info("Evaluation Summary")
        logger.info("=" * 50)
        for metric, value in metrics.items():
            logger.info(f"{metric}: {value:.4f}")
        
        self.metrics = metrics
        return metrics
    
    def compare_models(self, models_metrics: Dict[str, Dict]) -> pd.DataFrame:
        """
        Compare multiple models
        
        Args:
            models_metrics: Dict mapping model names to their metrics
            
        Returns:
            DataFrame with comparison
        """
        comparison_df = pd.DataFrame(models_metrics).T
        
        logger.info("\n" + "=" * 50)
        logger.info("Model Comparison")
        logger.info("=" * 50)
        print(comparison_df)
        
        return comparison_df


if __name__ == "__main__":
    # Example usage
    evaluator = ModelEvaluator()
    
    # Simulate some predictions
    true_ratings = np.array([5, 4, 3, 5, 4, 3, 2, 5])
    predicted_ratings = np.array([4.5, 4.2, 3.1, 4.8, 3.9, 3.2, 2.5, 4.7])
    
    # Calculate metrics
    rmse = evaluator.calculate_rmse(true_ratings, predicted_ratings)
    mae = evaluator.calculate_mae(true_ratings, predicted_ratings)
    
    # Simulate recommendations
    recommended = [1, 2, 3, 4, 5]
    relevant = [2, 4, 6, 8]
    
    precision = evaluator.precision_at_k(recommended, relevant, k=5)
    recall = evaluator.recall_at_k(recommended, relevant, k=5)
    
    print(f"\nPrecision@5: {precision:.4f}")
    print(f"Recall@5: {recall:.4f}")