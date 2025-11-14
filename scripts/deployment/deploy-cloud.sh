#!/bin/bash
# Deploy DoodleParty to cloud (Kubernetes)

set -e

echo "DoodleParty Cloud Deployment"
echo "============================"

NAMESPACE=${NAMESPACE:-"doodleparty"}
REGISTRY=${REGISTRY:-"docker.io"}
IMAGE_NAME=${IMAGE_NAME:-"doodleparty"}
TAG=${TAG:-"latest"}

echo "Namespace: $NAMESPACE"
echo "Image: $REGISTRY/$IMAGE_NAME:$TAG"

# Create namespace
echo "Creating namespace..."
kubectl create namespace $NAMESPACE --dry-run=client -o yaml | kubectl apply -f -

# Apply Kubernetes manifests
echo "Applying Kubernetes manifests..."
kubectl apply -f k8s/ -n $NAMESPACE

echo "Deployment complete!"
