"""
Database Configuration & Models
================================
PostgreSQL integration using SQLAlchemy ORM.

Why PostgreSQL?
- Stores recommendation logs (who asked for what) — useful for analytics
- Stores food items (so you can add/edit foods without retraining)
- Production-grade: used by Instagram, Spotify, Uber

Tables:
    foods              → All food items (mirrors CSV but editable)
    recommendation_logs → Tracks every API call for analytics
"""

import os
from datetime import datetime
from sqlalchemy import (
    create_engine, Column, Integer, String, Float, DateTime, Text
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# ── Database URL ──
# Local:  postgresql://postgres:password@localhost:5432/food_recommender
# Docker: postgresql://postgres:password@db:5432/food_recommender  (db = container name)
# EKS:    postgresql://user:pass@rds-endpoint:5432/food_recommender
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://postgres:obaidkhan123@localhost:5433/food_recommender"
)

# ── SQLAlchemy Setup ──
engine = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


# ── Database Models ──

class FoodDB(Base):
    """Food items table — mirrors the CSV dataset."""
    __tablename__ = "foods"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), unique=True, nullable=False, index=True)
    cuisine = Column(String(50), nullable=False)
    ingredients = Column(Text, nullable=False)
    spice_level = Column(String(20), nullable=False)
    diet_type = Column(String(20), nullable=False)
    rating = Column(Float, nullable=False)
    description = Column(Text)


class RecommendationLog(Base):
    """Logs every recommendation request — useful for analytics & monitoring."""
    __tablename__ = "recommendation_logs"

    id = Column(Integer, primary_key=True, index=True)
    input_food = Column(String(100), nullable=False)
    num_recommendations = Column(Integer, nullable=False)
    top_recommendation = Column(String(100))
    similarity_score = Column(Float)
    client_ip = Column(String(50))
    created_at = Column(DateTime, default=datetime.utcnow)


# ── Helper Functions ──

def get_db():
    """FastAPI dependency — provides a database session per request."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    """Create all tables if they don't exist."""
    Base.metadata.create_all(bind=engine)
    print("✅ Database tables created/verified")


def seed_foods_from_csv(csv_path: str = "data/food_dataset.csv"):
    """Load food data from CSV into PostgreSQL (runs once on first setup)."""
    import pandas as pd

    db = SessionLocal()
    try:
        # Skip if already seeded
        existing = db.query(FoodDB).count()
        if existing > 0:
            print(f"ℹ️  Database already has {existing} foods, skipping seed")
            return

        df = pd.read_csv(csv_path)
        for _, row in df.iterrows():
            food = FoodDB(
                name=row['name'],
                cuisine=row['cuisine'],
                ingredients=row['ingredients'],
                spice_level=row['spice_level'],
                diet_type=row['diet_type'],
                rating=row['rating'],
                description=row['description'],
            )
            db.add(food)

        db.commit()
        print(f"✅ Seeded {len(df)} foods into PostgreSQL")
    except Exception as e:
        db.rollback()
        print(f"⚠️  Database seed skipped (PostgreSQL may not be configured): {e}")
    finally:
        db.close()
