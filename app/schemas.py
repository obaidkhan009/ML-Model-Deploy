"""
Pydantic Schemas for Request/Response Validation
=================================================
These models ensure that:
- Incoming requests have the correct format
- API responses are consistent and well-documented
- Swagger/OpenAPI docs are auto-generated correctly
"""

from typing import List, Optional
from pydantic import BaseModel, Field


class RecommendRequest(BaseModel):
    """Request body for the /recommend endpoint."""
    food_name: str = Field(
        ...,
        description="Name of the food item to get recommendations for",
        json_schema_extra={"example": "Butter Chicken"}
    )
    num_recommendations: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of recommendations to return (1-20)"
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "food_name": "Butter Chicken",
                    "num_recommendations": 5
                }
            ]
        }
    }


class FoodItem(BaseModel):
    """A single food item in the recommendation response."""
    name: str = Field(..., description="Food name")
    cuisine: str = Field(..., description="Cuisine type (e.g., Indian, Thai)")
    ingredients: str = Field(..., description="Key ingredients")
    spice_level: str = Field(..., description="Spice level: mild, medium, hot")
    diet_type: str = Field(..., description="Diet type: vegan, vegetarian, non-veg")
    rating: float = Field(..., description="Average rating (1-5)")
    similarity_score: float = Field(
        ..., 
        description="How similar this food is to the input (0.0 to 1.0)"
    )


class RecommendResponse(BaseModel):
    """Response body for the /recommend endpoint."""
    input_food: str = Field(..., description="The food you asked recommendations for")
    num_results: int = Field(..., description="Number of recommendations returned")
    recommendations: List[FoodItem] = Field(
        ..., description="List of recommended food items"
    )


class HealthResponse(BaseModel):
    """Response for the /health endpoint."""
    status: str = Field(default="healthy")
    model_loaded: bool = Field(..., description="Whether the ML model is loaded")
    total_foods: int = Field(..., description="Number of foods in the database")


class FoodListResponse(BaseModel):
    """Response for the /foods endpoint."""
    total: int
    foods: List[str]
    cuisines: List[str]


class ErrorResponse(BaseModel):
    """Standard error response."""
    error: str
    available_foods: Optional[List[str]] = None


# ── AI (Ollama) Recommendation Schemas ──

class AIRecommendRequest(BaseModel):
    """Request body for the /recommend-ai endpoint (natural language)."""
    query: str = Field(
        ...,
        description="Natural language description of what you want to eat",
        json_schema_extra={"example": "I want something spicy with chicken, maybe Indian food"}
    )
    num_recommendations: int = Field(
        default=5, ge=1, le=20,
        description="Number of recommendations to return"
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "query": "I'm craving something spicy with chicken",
                    "num_recommendations": 5
                }
            ]
        }
    }


class AIRecommendResponse(BaseModel):
    """Response body for the /recommend-ai endpoint."""
    user_query: str = Field(..., description="The original natural language query")
    extracted_food: str = Field(..., description="Food name extracted by the LLM")
    llm_model: str = Field(..., description="Which LLM model was used")
    num_results: int = Field(..., description="Number of recommendations returned")
    recommendations: List[FoodItem] = Field(
        ..., description="List of recommended food items"
    )

