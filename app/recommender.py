"""
Food Recommender Engine
=======================
Core recommendation logic. Loads trained model artifacts and provides
recommendations using Cosine Similarity scores.

This class is loaded ONCE at API startup and reused across all requests
(singleton pattern) for fast inference.
"""

import os
import numpy as np
import pandas as pd
import joblib
from typing import List, Dict, Optional


class FoodRecommender:
    """
    Content-Based Food Recommendation Engine.
    
    How it works:
    1. On startup, loads pre-trained TF-IDF vectorizer and cosine similarity matrix
    2. When a user requests recommendations for a food:
       - Finds the food's index in the dataset
       - Looks up its row in the similarity matrix
       - Sorts by similarity score (descending)
       - Returns top N most similar foods
    
    Example:
        recommender = FoodRecommender("model/")
        results = recommender.get_recommendations("Butter Chicken", n=5)
    """
    
    def __init__(self, model_dir: str = "model"):
        """Load all model artifacts from disk."""
        self.model_dir = model_dir
        self.food_data: Optional[pd.DataFrame] = None
        self.similarity_matrix: Optional[np.ndarray] = None
        self.is_loaded = False
        
        self._load_model()
    
    def _load_model(self):
        """Load pre-trained model artifacts."""
        try:
            self.food_data = joblib.load(
                os.path.join(self.model_dir, 'food_data.pkl')
            )
            self.similarity_matrix = joblib.load(
                os.path.join(self.model_dir, 'similarity_matrix.pkl')
            )
            # We don't need the vectorizer for inference — similarity is precomputed
            self.is_loaded = True
            print(f"✅ Model loaded: {len(self.food_data)} foods, "
                  f"similarity matrix {self.similarity_matrix.shape}")
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            self.is_loaded = False
    
    @property
    def total_foods(self) -> int:
        """Total number of foods in the database."""
        return len(self.food_data) if self.food_data is not None else 0
    
    def get_all_food_names(self) -> List[str]:
        """Return list of all food names."""
        if self.food_data is None:
            return []
        return self.food_data['name'].tolist()
    
    def get_all_cuisines(self) -> List[str]:
        """Return list of all unique cuisines."""
        if self.food_data is None:
            return []
        return sorted(self.food_data['cuisine'].unique().tolist())
    
    def food_exists(self, food_name: str) -> bool:
        """Check if a food exists in the database (case-insensitive)."""
        if self.food_data is None:
            return False
        return food_name.lower() in self.food_data['name'].str.lower().values
    
    def _find_food_index(self, food_name: str) -> int:
        """Find the index of a food item (case-insensitive)."""
        mask = self.food_data['name'].str.lower() == food_name.lower()
        matches = self.food_data[mask]
        if len(matches) == 0:
            raise ValueError(f"Food '{food_name}' not found")
        return matches.index[0]
    
    def get_recommendations(
        self, 
        food_name: str, 
        n: int = 5,
        diet_filter: Optional[str] = None
    ) -> List[Dict]:
        """
        Get top-N food recommendations similar to the given food.
        
        Args:
            food_name: Name of the food to find similar items for
            n: Number of recommendations to return
            diet_filter: Optional filter (vegan, vegetarian, non-veg)
            
        Returns:
            List of dicts with food details + similarity scores
            
        Algorithm:
            1. Find food's index in the dataset
            2. Get its row from the similarity matrix (scores against ALL other foods)
            3. Sort by similarity score descending
            4. Skip the food itself (score = 1.0)
            5. Return top N results
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded")
        
        # Find the food's index
        idx = self._find_food_index(food_name)
        
        # Get similarity scores for this food against all others
        sim_scores = list(enumerate(self.similarity_matrix[idx]))
        
        # Sort by similarity (highest first)
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        
        # Skip the first one (it's the food itself, similarity = 1.0)
        sim_scores = sim_scores[1:]
        
        # Apply diet filter if specified
        if diet_filter:
            sim_scores = [
                (i, score) for i, score in sim_scores
                if self.food_data.iloc[i]['diet_type'].lower() == diet_filter.lower()
            ]
        
        # Take top N
        top_scores = sim_scores[:n]
        
        # Build response
        recommendations = []
        for i, score in top_scores:
            food = self.food_data.iloc[i]
            recommendations.append({
                'name': food['name'],
                'cuisine': food['cuisine'],
                'ingredients': food['ingredients'],
                'spice_level': food['spice_level'],
                'diet_type': food['diet_type'],
                'rating': float(food['rating']),
                'similarity_score': round(float(score), 4),
            })
        
        return recommendations
