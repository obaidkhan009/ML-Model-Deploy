#!/bin/bash
# ==========================================
# EKS Cluster Setup Script
# ==========================================
# Creates an EKS cluster using eksctl.
#
# Prerequisites:
#   - AWS CLI configured (aws configure)
#   - eksctl installed: brew install eksctl
#   - kubectl installed: brew install kubectl
#
# ⚠️ WARNING: This creates AWS resources that cost money!
#   - EKS control plane: ~$0.10/hour
#   - 2x t3.medium nodes: ~$0.084/hour
#   - Total: ~$4.40/day
#
# Usage:
#   chmod +x scripts/setup-eks.sh
#   ./scripts/setup-eks.sh
# ==========================================

set -e

# ── Configuration (CHANGE THESE) ──
CLUSTER_NAME="ml-deploy-cluster"
AWS_REGION="us-east-1"
NODE_TYPE="t3.medium"       # Good balance of CPU/memory for ML workloads
NODE_COUNT=2                # Start with 2 nodes
MIN_NODES=2
MAX_NODES=4

echo "🚀 Setting up EKS Cluster: ${CLUSTER_NAME}"
echo "   Region: ${AWS_REGION}"
echo "   Node type: ${NODE_TYPE}"
echo "   Nodes: ${NODE_COUNT} (min: ${MIN_NODES}, max: ${MAX_NODES})"
echo ""
echo "⏳ This will take 15-20 minutes..."
echo ""

# Step 1: Create EKS cluster with managed node group
eksctl create cluster \
    --name ${CLUSTER_NAME} \
    --region ${AWS_REGION} \
    --version 1.28 \
    --nodegroup-name ml-workers \
    --node-type ${NODE_TYPE} \
    --nodes ${NODE_COUNT} \
    --nodes-min ${MIN_NODES} \
    --nodes-max ${MAX_NODES} \
    --managed \
    --asg-access

# Step 2: Verify cluster is running
echo ""
echo "🔍 Verifying cluster..."
kubectl get nodes
kubectl cluster-info

# Step 3: Install metrics-server (required for HPA)
echo ""
echo "📊 Installing metrics-server (required for autoscaling)..."
kubectl apply -f https://github.com/kubernetes-sigs/metrics-server/releases/latest/download/components.yaml

# Step 4: Create the namespace
echo ""
echo "📁 Creating ml-serving namespace..."
kubectl apply -f k8s/namespace.yaml

echo ""
echo "✅ EKS Cluster is ready!"
echo ""
echo "📋 Next steps:"
echo "   1. Push your image to ECR: ./scripts/setup-ecr.sh"
echo "   2. Update the image URI in k8s/deployment.yaml"
echo "   3. Deploy: kubectl apply -f k8s/"
echo "   4. Check pods: kubectl get pods -n ml-serving"
echo "   5. Get endpoint: kubectl get ingress -n ml-serving"
echo ""
echo "💰 REMINDER: To avoid charges, delete the cluster when done:"
echo "   eksctl delete cluster --name ${CLUSTER_NAME} --region ${AWS_REGION}"
