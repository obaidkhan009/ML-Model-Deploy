#!/bin/bash
# ==========================================
# ECR Setup Script
# ==========================================
# Creates an ECR repository and pushes your Docker image.
#
# Prerequisites:
#   - AWS CLI installed and configured (aws configure)
#   - Docker installed and running
#
# Usage:
#   chmod +x scripts/setup-ecr.sh
#   ./scripts/setup-ecr.sh
# ==========================================

set -e  # Exit on any error

# ── Configuration (CHANGE THESE) ──
AWS_REGION="us-east-1"
ECR_REPO_NAME="food-recommender"
IMAGE_TAG="latest"

echo "🚀 Setting up ECR for ${ECR_REPO_NAME}..."

# Step 1: Create ECR repository (if it doesn't exist)
echo "📦 Creating ECR repository..."
aws ecr create-repository \
    --repository-name ${ECR_REPO_NAME} \
    --region ${AWS_REGION} \
    --image-scanning-configuration scanOnPush=true \
    2>/dev/null || echo "   Repository already exists, skipping..."

# Step 2: Get the ECR registry URI
AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
ECR_URI="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"
FULL_IMAGE="${ECR_URI}/${ECR_REPO_NAME}:${IMAGE_TAG}"

echo "📍 ECR URI: ${ECR_URI}"
echo "🏷️  Full image: ${FULL_IMAGE}"

# Step 3: Authenticate Docker with ECR
echo "🔐 Authenticating Docker with ECR..."
aws ecr get-login-password --region ${AWS_REGION} | \
    docker login --username AWS --password-stdin ${ECR_URI}

# Step 4: Build the Docker image
echo "🔨 Building Docker image..."
docker build -t ${ECR_REPO_NAME}:${IMAGE_TAG} .

# Step 5: Tag the image for ECR
echo "🏷️  Tagging image..."
docker tag ${ECR_REPO_NAME}:${IMAGE_TAG} ${FULL_IMAGE}

# Step 6: Push to ECR
echo "⬆️  Pushing image to ECR..."
docker push ${FULL_IMAGE}

echo ""
echo "✅ SUCCESS! Image pushed to ECR."
echo "   Image URI: ${FULL_IMAGE}"
echo ""
echo "📋 Next steps:"
echo "   1. Update k8s/deployment.yaml with this image URI"
echo "   2. Run: ./scripts/setup-eks.sh (if cluster doesn't exist)"
echo "   3. Run: kubectl apply -f k8s/"
