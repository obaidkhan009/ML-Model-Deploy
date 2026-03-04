"""
Food Recommendation API — FastAPI Application
==============================================
Production-ready REST API for serving food recommendations.

Endpoints:
    GET  /          → Welcome message
    GET  /health    → Liveness probe (for Kubernetes)
    GET  /foods     → List all available foods
    POST /recommend → Get food recommendations
    GET  /metrics   → Prometheus metrics (auto-generated)

Architecture:
    Client → AWS Load Balancer → K8s Service → This API → Pre-loaded Model (.pkl)
                                                        → PostgreSQL (logs & analytics)
"""

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from prometheus_fastapi_instrumentator import Instrumentator

from app.schemas import (
    RecommendRequest,
    RecommendResponse,
    FoodItem,
    HealthResponse,
    FoodListResponse,
    ErrorResponse,
)
from app.recommender import FoodRecommender


# ──────────────────────────────────────────────
# Initialize FastAPI App
# ──────────────────────────────────────────────
app = FastAPI(
    title="🍽️ Food Recommendation API",
    description=(
        "A Content-Based Filtering recommendation engine that suggests similar foods "
        "based on cuisine, ingredients, spice level, and dietary type. "
        "Built with Scikit-learn (TF-IDF + Cosine Similarity) and deployed on AWS EKS."
    ),
    version="1.0.0",
    docs_url="/docs",       # Swagger UI
    redoc_url="/redoc",      # ReDoc UI
)

# CORS — allow all origins (for development / testing)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Prometheus metrics — auto instruments all endpoints
# Exposes /metrics endpoint for Prometheus to scrape
Instrumentator().instrument(app).expose(app)


# ──────────────────────────────────────────────
# Load Model on Startup
# ──────────────────────────────────────────────
recommender = FoodRecommender(model_dir="model")


# ──────────────────────────────────────────────
# Database Initialization (Optional — works without DB too)
# ──────────────────────────────────────────────
db_available = False
try:
    from app.database import init_db, seed_foods_from_csv, SessionLocal, RecommendationLog
    init_db()
    seed_foods_from_csv()
    db_available = True
    print("✅ PostgreSQL connected")
except Exception as e:
    print(f"⚠️  PostgreSQL not available, running without database: {e}")
    print("   (Recommendations still work — DB is only for logging)")


# ──────────────────────────────────────────────
# Endpoints
# ──────────────────────────────────────────────

@app.get("/", tags=["General"])
def root():
    """Welcome endpoint."""
    return {
        "service": "Food Recommendation API",
        "version": "1.0.0",
        "database": "connected" if db_available else "not configured",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
def health_check():
    """
    Health check endpoint — used by Kubernetes liveness/readiness probes.
    
    Kubernetes calls this endpoint every 10 seconds.
    If it fails 3 times in a row, K8s restarts the pod.
    """
    return HealthResponse(
        status="healthy" if recommender.is_loaded else "unhealthy",
        model_loaded=recommender.is_loaded,
        total_foods=recommender.total_foods,
    )


@app.get("/foods", response_model=FoodListResponse, tags=["Foods"])
def list_foods():
    """List all available food items and cuisines in the database."""
    return FoodListResponse(
        total=recommender.total_foods,
        foods=recommender.get_all_food_names(),
        cuisines=recommender.get_all_cuisines(),
    )


@app.post(
    "/recommend",
    response_model=RecommendResponse,
    responses={404: {"model": ErrorResponse}},
    tags=["Recommendations"],
)
def get_recommendations(request: RecommendRequest, req: Request = None):
    """
    Get food recommendations based on a food item you like.
    
    **How it works:**
    1. You tell us a food you like (e.g., "Butter Chicken")
    2. Our ML model finds the most similar foods using Cosine Similarity
    3. Returns top N recommendations sorted by similarity score
    
    **Example request:**
    ```json
    {
        "food_name": "Butter Chicken",
        "num_recommendations": 5
    }
    ```
    """
    # Check if model is loaded
    if not recommender.is_loaded:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please try again later."
        )
    
    # Check if food exists
    if not recommender.food_exists(request.food_name):
        raise HTTPException(
            status_code=404,
            detail={
                "error": f"Food '{request.food_name}' not found in our database.",
                "available_foods": recommender.get_all_food_names(),
            }
        )
    
    # Get recommendations from the ML model
    results = recommender.get_recommendations(
        food_name=request.food_name,
        n=request.num_recommendations,
    )
    
    # Log to PostgreSQL (if database is available)
    if db_available:
        try:
            db = SessionLocal()
            log_entry = RecommendationLog(
                input_food=request.food_name,
                num_recommendations=request.num_recommendations,
                top_recommendation=results[0]['name'] if results else None,
                similarity_score=results[0]['similarity_score'] if results else None,
                client_ip=req.client.host if req else None,
            )
            db.add(log_entry)
            db.commit()
            db.close()
        except Exception:
            pass  # Don't fail the request if logging fails
    
    # Build response
    recommendations = [FoodItem(**item) for item in results]
    
    return RecommendResponse(
        input_food=request.food_name,
        num_results=len(recommendations),
        recommendations=recommendations,
    )
