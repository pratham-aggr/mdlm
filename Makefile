.PHONY: help build push run run-local deploy-job deploy-service logs clean

DOCKER_IMAGE ?= mdlm:latest
REGISTRY ?= registry.nrp.ai
USERNAME ?= your-username
KUBE_NAMESPACE ?= ucsd-qswang-lab

help:
	@echo "MDLM Makefile Commands"
	@echo ""
	@echo "Local Development:"
	@echo "  make build          Build Docker image locally"
	@echo "  make run-local      Run with docker-compose (includes GPU)"
	@echo "  make logs           View docker-compose logs"
	@echo "  make shell          Open bash in running container"
	@echo ""
	@echo "NRP.ai Deployment:"
	@echo "  make push           Push image to registry (requires REGISTRY and USERNAME)"
	@echo "  make deploy-job     Deploy as Kubernetes Job"
	@echo "  make deploy-service Deploy as Kubernetes Service"
	@echo "  make k8s-logs       View Kubernetes logs"
	@echo "  make k8s-status     Check pod status"
	@echo ""
	@echo "Cleanup:"
	@echo "  make clean          Stop docker-compose and remove containers"
	@echo "  make k8s-clean      Delete Kubernetes namespace"
	@echo ""

build:
	docker build -t $(DOCKER_IMAGE) .

push:
	docker tag $(DOCKER_IMAGE) $(REGISTRY)/$(USERNAME)/$(DOCKER_IMAGE)
	docker push $(REGISTRY)/$(USERNAME)/$(DOCKER_IMAGE)

run-local:
	@echo "Starting Docker Compose..."
	docker-compose up --build

run-local-detached:
	docker-compose up -d --build

logs:
	docker-compose logs -f

shell:
	docker exec -it mdlm-experiment bash

stop-local:
	docker-compose down

deploy-job:
	@echo "Creating namespace and secrets..."
	kubectl create namespace $(KUBE_NAMESPACE) --dry-run=client -o yaml | kubectl apply -f -
	kubectl create secret generic mdlm-secrets --from-literal=hf-token="${HF_TOKEN}" \
		-n $(KUBE_NAMESPACE) --dry-run=client -o yaml | kubectl apply -f -
	@echo "Deploying Job..."
	kubectl apply -f k8s/job.yaml -n $(KUBE_NAMESPACE)
	@echo "✓ Job deployed. Monitor with:"
	@echo "  kubectl logs -f job/mdlm-experiment-job -n $(KUBE_NAMESPACE)"

deploy-service:
	@echo "Creating namespace and secrets..."
	kubectl create namespace $(KUBE_NAMESPACE) --dry-run=client -o yaml | kubectl apply -f -
	kubectl create secret generic mdlm-secrets --from-literal=hf-token="${HF_TOKEN}" \
		-n $(KUBE_NAMESPACE) --dry-run=client -o yaml | kubectl apply -f -
	@echo "Deploying Service..."
	kubectl apply -f k8s/deployment.yaml -n $(KUBE_NAMESPACE)
	@echo "✓ Deployment created. Monitor with:"
	@echo "  kubectl logs -f deployment/mdlm-app -n $(KUBE_NAMESPACE)"

k8s-logs:
	kubectl logs -f -n $(KUBE_NAMESPACE) $$(kubectl get pod -n $(KUBE_NAMESPACE) -o name | head -1)

k8s-status:
	@echo "Pods:"
	@kubectl get pods -n $(KUBE_NAMESPACE)
	@echo ""
	@echo "PVCs:"
	@kubectl get pvc -n $(KUBE_NAMESPACE)
	@echo ""
	@echo "Secrets:"
	@kubectl get secrets -n $(KUBE_NAMESPACE)

k8s-describe:
	kubectl describe pod $$(kubectl get pod -n $(KUBE_NAMESPACE) -o name | head -1) -n $(KUBE_NAMESPACE)

k8s-shell:
	kubectl exec -it $$(kubectl get pod -n $(KUBE_NAMESPACE) -o name | head -1) -n $(KUBE_NAMESPACE) -- bash

k8s-clean:
	kubectl delete namespace $(KUBE_NAMESPACE)

clean: stop-local
	docker system prune -f

all: build run-local

.DEFAULT_GOAL := help
