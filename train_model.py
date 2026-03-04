"""
Food Recommendation Model Training Script
==========================================
Trains a Content-Based Filtering model using TF-IDF and Cosine Similarity.

How it works:
1. Loads food dataset (CSV)
2. Combines text features (cuisine, ingredients, spice_level, diet_type) into one string
3. Converts text to numerical vectors using TF-IDF
4. Computes similarity between all food pairs using Cosine Similarity
5. Saves trained artifacts as .pkl files for the API to load

Usage:
    python train_model.py
"""

import os
import pandas as pd
import numpy as np
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def load_data(filepath: str) -> pd.DataFrame:
    """Load the food dataset from CSV."""
    df = pd.read_csv(filepath)
    print(f"✅ Loaded {len(df)} food items from {filepath}")
    print(f"   Cuisines: {df['cuisine'].nunique()} unique")
    print(f"   Diet types: {df['diet_type'].unique().tolist()}")
    return df


def create_feature_text(df: pd.DataFrame) -> pd.Series:
    """
    Combine relevant columns into a single text string per food item.
    
    This is the KEY step — we merge all the features that define a food
    so TF-IDF can learn what makes each food unique.
    
    Example output for Butter Chicken:
    "Indian chicken cream butter tomato garam_masala fenugreek ginger garlic medium non-veg"
    """
    feature_text = (
        df['cuisine'] + ' ' +
        df['ingredients'] + ' ' +
        df['spice_level'] + ' ' +
        df['diet_type'] + ' ' +
        df['description'].str.lower()
    )
    return feature_text


def train_model(df: pd.DataFrame, feature_text: pd.Series):
    """
    Train the recommendation model.
    
    Step 1: TF-IDF Vectorizer
    - Converts each food's text features into a numerical vector
    - Words that appear everywhere (like "garlic") get LOW weight
    - Unique words (like "tandoori") get HIGH weight
    - This helps distinguish foods from each other
    
    Step 2: Cosine Similarity
    - Compares every food vector against every other food vector
    - Returns a score from 0.0 (totally different) to 1.0 (identical)
    - Creates an NxN matrix (50 foods = 50x50 = 2500 similarity scores)
    """
    # Step 1: Fit TF-IDF
    tfidf = TfidfVectorizer(
        stop_words='english',    # Remove common English words (the, is, at...)
        max_features=5000,       # Keep top 5000 most important terms
        ngram_range=(1, 2),      # Consider single words AND two-word phrases
    )
    tfidf_matrix = tfidf.fit_transform(feature_text)
    
    print(f"\n📊 TF-IDF Matrix Shape: {tfidf_matrix.shape}")
    print(f"   {tfidf_matrix.shape[0]} food items × {tfidf_matrix.shape[1]} unique features")
    
    # Step 2: Compute Cosine Similarity
    similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)
    
    print(f"\n🔗 Similarity Matrix Shape: {similarity_matrix.shape}")
    print(f"   {similarity_matrix.shape[0]}×{similarity_matrix.shape[1]} pairwise similarities")
    
    return tfidf, tfidf_matrix, similarity_matrix


def show_sample_recommendations(df: pd.DataFrame, similarity_matrix: np.ndarray):
    """Display sample recommendations to verify the model works."""
    sample_foods = ["Butter Chicken", "Sushi Roll", "Margherita Pizza", "Pad Thai"]
    
    print("\n" + "=" * 60)
    print("🍽️  SAMPLE RECOMMENDATIONS")
    print("=" * 60)
    
    for food_name in sample_foods:
        if food_name not in df['name'].values:
            continue
            
        idx = df[df['name'] == food_name].index[0]
        # Get similarity scores, sort descending, skip self (index 0)
        sim_scores = list(enumerate(similarity_matrix[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:6]  # Top 5 (skip self)
        
        print(f"\n🔹 If you like '{food_name}', try:")
        for rank, (i, score) in enumerate(sim_scores, 1):
            print(f"   {rank}. {df.iloc[i]['name']} ({df.iloc[i]['cuisine']}) "
                  f"— similarity: {score:.3f}")


def save_artifacts(df: pd.DataFrame, tfidf, similarity_matrix: np.ndarray, output_dir: str):
    """Save all model artifacts for the API to load at startup."""
    os.makedirs(output_dir, exist_ok=True)
    
    joblib.dump(tfidf, os.path.join(output_dir, 'tfidf_vectorizer.pkl'))
    joblib.dump(similarity_matrix, os.path.join(output_dir, 'similarity_matrix.pkl'))
    joblib.dump(df, os.path.join(output_dir, 'food_data.pkl'))
    
    print(f"\n💾 Saved model artifacts to {output_dir}/")
    for f in os.listdir(output_dir):
        size = os.path.getsize(os.path.join(output_dir, f))
        print(f"   {f} ({size:,} bytes)")


def main():
    print("🚀 Food Recommendation Model Training")
    print("=" * 50)
    
    # 1. Load data
    df = load_data('data/food_dataset.csv')
    
    # 2. Create combined feature text
    feature_text = create_feature_text(df)
    
    # 3. Train model (TF-IDF + Cosine Similarity)
    tfidf, tfidf_matrix, similarity_matrix = train_model(df, feature_text)
    
    # 4. Show sample recommendations to verify
    show_sample_recommendations(df, similarity_matrix)
    
    # 5. Save artifacts
    save_artifacts(df, tfidf, similarity_matrix, 'model')
    
    print("\n✅ Training complete! Model ready for deployment.")


if __name__ == '__main__':
    main()
