"""
LLM Service — Ollama Integration
=================================
Connects to Ollama to process natural language food queries.

How it works:
1. User sends a natural language query: "I want spicy chicken, maybe Indian"
2. We send a prompt to Ollama (llama3.2:1b) asking it to extract a food name
3. Ollama returns: "Tandoori Chicken"
4. We pass that to our existing recommender engine
5. User gets personalized recommendations

Ollama runs as a separate service:
  - Local dev:  http://localhost:11434
  - Docker:     http://ollama:11434
  - Kubernetes: http://ollama-service:11434
"""

import os
import httpx
import json
from typing import Optional

# Ollama API endpoint
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:1b")

# The prompt template — tells the LLM exactly what to do
EXTRACTION_PROMPT = """You are a food recommendation assistant. Given a user's description of what they want to eat, extract the SINGLE most relevant food name from this list:

{food_list}

Rules:
- Return ONLY the food name, nothing else
- Pick the food that best matches the user's description
- If the description mentions a cuisine (Indian, Thai, etc.), prefer foods from that cuisine
- If the description mentions ingredients, match foods with those ingredients
- If no good match exists, return the closest option

User's request: "{user_query}"

Food name:"""


async def extract_food_from_query(
    user_query: str,
    available_foods: list[str],
) -> Optional[str]:
    """
    Use Ollama LLM to extract a food name from a natural language query.
    
    Args:
        user_query: Natural language like "I want something spicy with chicken"
        available_foods: List of all food names in our database
        
    Returns:
        Extracted food name, or None if LLM is unavailable
    """
    food_list = ", ".join(available_foods)
    prompt = EXTRACTION_PROMPT.format(
        food_list=food_list,
        user_query=user_query,
    )
    
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{OLLAMA_URL}/api/generate",
                json={
                    "model": OLLAMA_MODEL,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.1,    # Low temp = more deterministic
                        "num_predict": 20,      # Short response (just a food name)
                    }
                },
            )
            
            if response.status_code == 200:
                result = response.json()
                food_name = result.get("response", "").strip()
                # Clean up the response (remove quotes, periods, etc.)
                food_name = food_name.strip('"\'.,!').strip()
                return food_name
            else:
                print(f"⚠️ Ollama returned status {response.status_code}")
                return None
                
    except httpx.ConnectError:
        print(f"⚠️ Cannot connect to Ollama at {OLLAMA_URL}")
        return None
    except Exception as e:
        print(f"⚠️ Ollama error: {e}")
        return None


async def check_ollama_health() -> dict:
    """Check if Ollama is running and what models are available."""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{OLLAMA_URL}/api/tags")
            if response.status_code == 200:
                models = response.json().get("models", [])
                model_names = [m["name"] for m in models]
                return {
                    "status": "connected",
                    "url": OLLAMA_URL,
                    "model": OLLAMA_MODEL,
                    "available_models": model_names,
                }
            return {"status": "error", "url": OLLAMA_URL}
    except Exception:
        return {"status": "disconnected", "url": OLLAMA_URL}
