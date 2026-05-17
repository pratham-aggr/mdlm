#!/bin/bash
# Quick deployment script for NRP.ai

set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${GREEN}=== MDLM Deployment to NRP.ai ===${NC}\n"

# Function to prompt user
prompt_user() {
    read -p "$(echo -e ${YELLOW}$1${NC})" response
    echo "$response"
}

# Check prerequisites
echo -e "${YELLOW}Checking prerequisites...${NC}"
command -v docker &> /dev/null || { echo -e "${RED}Docker not found${NC}"; exit 1; }
command -v kubectl &> /dev/null || { echo -e "${RED}kubectl not found${NC}"; exit 1; }
echo -e "${GREEN}✓ Prerequisites met${NC}\n"

# Get user inputs
HF_TOKEN=$(prompt_user "Enter your HuggingFace token: ")
REGISTRY=$(prompt_user "Enter registry URL (e.g., registry.nrp.ai or docker.io): ")
USERNAME=$(prompt_user "Enter your registry username: ")
IMAGE_TAG=$(prompt_user "Enter image tag (default: latest): ")
IMAGE_TAG=${IMAGE_TAG:-latest}

KUBECONFIG_PATH=$(prompt_user "Enter path to kubeconfig (default: ~/.kube/config): ")
KUBECONFIG_PATH=${KUBECONFIG_PATH:-~/.kube/config}

# Validate inputs
if [ -z "$HF_TOKEN" ]; then
    echo -e "${RED}✗ HF_TOKEN is required${NC}"
    exit 1
fi

if [ ! -f "$KUBECONFIG_PATH" ]; then
    echo -e "${RED}✗ kubeconfig not found at $KUBECONFIG_PATH${NC}"
    exit 1
fi

export KUBECONFIG="$KUBECONFIG_PATH"

# Build Docker image
echo -e "\n${YELLOW}Building Docker image...${NC}"
IMAGE_NAME="${REGISTRY}/${USERNAME}/mdlm:${IMAGE_TAG}"
docker build -t "$IMAGE_NAME" .
echo -e "${GREEN}✓ Image built: $IMAGE_NAME${NC}"

# Push to registry
echo -e "\n${YELLOW}Pushing image to registry...${NC}"
docker push "$IMAGE_NAME"
echo -e "${GREEN}✓ Image pushed${NC}"

# Verify cluster access
echo -e "\n${YELLOW}Verifying NRP.ai cluster access...${NC}"
if ! kubectl cluster-info &> /dev/null; then
    echo -e "${RED}✗ Cannot connect to Kubernetes cluster${NC}"
    echo "Make sure KUBECONFIG is set correctly"
    exit 1
fi
echo -e "${GREEN}✓ Cluster access verified${NC}"

# Use provided namespace
NAMESPACE="ucsd-qswang-lab"
echo -e "\n${YELLOW}Setting up secrets in namespace: $NAMESPACE${NC}"
# Note: Namespace should already exist on NRP.ai; we just create the secret

# Create secret
kubectl create secret generic mdlm-secrets \
  --from-literal=hf-token="$HF_TOKEN" \
  -n "$NAMESPACE" --dry-run=client -o yaml | kubectl apply -f -
echo -e "${GREEN}✓ Namespace and secrets created${NC}"

# Update image in manifests
echo -e "\n${YELLOW}Updating Kubernetes manifests...${NC}"
sed -i '' "s|image:.*|image: $IMAGE_NAME|g" k8s/job.yaml k8s/deployment.yaml

# Deploy
DEPLOY_TYPE=$(prompt_user "Deploy as [1] Job (one-time experiment) or [2] Service (long-running): ")

if [ "$DEPLOY_TYPE" = "1" ]; then
    echo -e "\n${YELLOW}Deploying as Job...${NC}"
    kubectl apply -f k8s/job.yaml -n "$NAMESPACE"

    # Get job name
    JOB_NAME=$(kubectl get jobs -n "$NAMESPACE" -o jsonpath='{.items[0].metadata.name}')
    echo -e "${GREEN}✓ Job deployed: $JOB_NAME${NC}"

    # Show monitoring command
    echo -e "\n${GREEN}Monitor with:${NC}"
    echo "  kubectl logs -f job/$JOB_NAME -n $NAMESPACE"
else
    echo -e "\n${YELLOW}Deploying as Service...${NC}"
    kubectl apply -f k8s/deployment.yaml -n "$NAMESPACE"

    POD_NAME=$(kubectl get pods -n "$NAMESPACE" -o jsonpath='{.items[0].metadata.name}')
    echo -e "${GREEN}✓ Deployment created${NC}"

    echo -e "\n${GREEN}Monitor with:${NC}"
    echo "  kubectl logs -f pod/$POD_NAME -n $NAMESPACE"
fi

echo -e "\n${GREEN}=== Deployment Complete ===${NC}"
echo -e "\nUseful commands:"
echo "  kubectl get pods -n $NAMESPACE"
echo "  kubectl describe pod <pod-name> -n $NAMESPACE"
echo "  kubectl exec -it <pod-name> -n $NAMESPACE -- /bin/bash"
echo "  kubectl logs -f <pod-name> -n $NAMESPACE"
echo "  kubectl delete namespace $NAMESPACE"
