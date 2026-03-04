# 🍽️ Food Recommendation API — AWS EKS Deployment

A production-grade **Food Recommendation** microservice built with **Scikit-learn** and **FastAPI**, deployed on **AWS EKS** (Kubernetes) with autoscaling, CI/CD, and monitoring.

## 🏗️ Architecture

```
User Request
     ↓
AWS Load Balancer (ALB)
     ↓
Kubernetes Service
     ↓
ML Model Pods (FastAPI) × 2-10 replicas
     ↓
Saved Model (.pkl)
```

## 🧠 ML Model

| Component | Technology |
|---|---|
| Algorithm | Content-Based Filtering |
| Vectorizer | TF-IDF (sklearn) |
| Similarity | Cosine Similarity |
| Dataset | 50 foods across 10 cuisines |
| Model Format | `.pkl` (joblib) |

### How Recommendations Work
1. Food features (cuisine, ingredients, spice level) → **TF-IDF** converts to numerical vectors
2. **Cosine Similarity** measures how similar any two foods are (0.0 → 1.0)
3. API returns the top N most similar foods

## 📁 Project Structure

```
ML-Model-Deploy/
├── app/
│   ├── main.py              # FastAPI application
│   ├── schemas.py            # Pydantic request/response models
│   └── recommender.py        # ML recommendation engine
├── data/
│   └── food_dataset.csv      # Food items dataset
├── model/                     # Trained model artifacts (generated)
├── k8s/                       # Kubernetes manifests
│   ├── namespace.yaml
│   ├── deployment.yaml
│   ├── service.yaml
│   ├── hpa.yaml
│   └── ingress.yaml
├── helm/food-recommender/     # Helm chart
├── scripts/
│   ├── setup-ecr.sh           # Push image to ECR
│   └── setup-eks.sh           # Create EKS cluster
├── .github/workflows/
│   └── deploy.yml             # CI/CD pipeline
├── train_model.py             # Model training script
├── Dockerfile
├── requirements.txt
└── README.md
```

## 🚀 Quick Start

### 1. Setup Python Environment
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Train the Model
```bash
python train_model.py
```
This creates `model/` directory with:
- `tfidf_vectorizer.pkl` — The trained TF-IDF vectorizer
- `similarity_matrix.pkl` — Pre-computed similarity scores
- `food_data.pkl` — Food dataset

### 3. Run API Locally
```bash
uvicorn app.main:app --reload
```

### 4. Test the API
```bash
# Health check
curl http://localhost:8000/health

# List all foods
curl http://localhost:8000/foods

# Get recommendations
curl -X POST http://localhost:8000/recommend \
  -H "Content-Type: application/json" \
  -d '{"food_name": "Butter Chicken", "num_recommendations": 5}'
```

### 5. View API Docs
Open http://localhost:8000/docs for interactive Swagger UI.

---

## 🐳 Docker Deployment

```bash
# Build image
docker build -t food-recommender .

# Run container
docker run -p 8000:8000 food-recommender

# Test
curl http://localhost:8000/health
```

---

## ☁️ AWS EKS Deployment

### Prerequisites
- AWS CLI configured (`aws configure`)
- `eksctl` installed (`brew install eksctl`)
- `kubectl` installed (`brew install kubectl`)
- Docker running

### Step 1: Create EKS Cluster
```bash
chmod +x scripts/setup-eks.sh
./scripts/setup-eks.sh
```
⏳ Takes ~15-20 minutes. Creates a 2-node cluster.

### Step 2: Push Image to ECR
```bash
chmod +x scripts/setup-ecr.sh
./scripts/setup-ecr.sh
```

### Step 3: Update Image URI
Edit `k8s/deployment.yaml` and replace the image placeholder:
```yaml
image: <YOUR_ACCOUNT_ID>.dkr.ecr.<REGION>.amazonaws.com/food-recommender:latest
```

### Step 4: Deploy to Kubernetes
```bash
kubectl apply -f k8s/
```

### Step 5: Verify
```bash
kubectl get pods -n ml-serving          # Should show 2 running pods
kubectl get svc -n ml-serving           # Shows the service
kubectl get hpa -n ml-serving           # Shows autoscaler status
kubectl get ingress -n ml-serving       # Shows external ALB URL
```

### Step 6: Test via ALB
```bash
ALB_URL=$(kubectl get ingress -n ml-serving -o jsonpath='{.items[0].status.loadBalancer.ingress[0].hostname}')
curl http://${ALB_URL}/health
```

---

## 📦 Helm Deployment (Alternative)

```bash
# Install
helm install food-recommender ./helm/food-recommender -n ml-serving

# Upgrade
helm upgrade food-recommender ./helm/food-recommender -n ml-serving

# Uninstall
helm uninstall food-recommender -n ml-serving
```

---

## 📊 Monitoring

The API auto-exposes Prometheus metrics at `/metrics`:
```bash
curl http://localhost:8000/metrics
```

Metrics include:
- Request count per endpoint
- Request duration (latency)
- Response size
- Active requests

---

## 🧹 Cleanup (Stop AWS Charges)

```bash
# Delete K8s resources
kubectl delete -f k8s/

# Delete EKS cluster (this takes ~10 minutes)
eksctl delete cluster --name ml-deploy-cluster --region us-east-1

# Delete ECR repository
aws ecr delete-repository --repository-name food-recommender --force --region us-east-1
```

---

## 📄 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Welcome message |
| GET | `/health` | Health check (K8s probes) |
| GET | `/foods` | List all foods + cuisines |
| POST | `/recommend` | Get food recommendations |
| GET | `/metrics` | Prometheus metrics |
| GET | `/docs` | Swagger UI |

### Example Request
```json
POST /recommend
{
  "food_name": "Butter Chicken",
  "num_recommendations": 5
}
```

### Example Response
```json
{
  "input_food": "Butter Chicken",
  "num_results": 5,
  "recommendations": [
    {
      "name": "Chicken Tikka Masala",
      "cuisine": "Indian",
      "ingredients": "chicken yogurt tomato cream tikka_spice ginger garlic onion",
      "spice_level": "medium",
      "diet_type": "non-veg",
      "rating": 4.7,
      "similarity_score": 0.6234
    }
  ]
}
```
