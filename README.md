Assignment2: Deploying PyTorch ML Workloads with Docker & Kubernetes
``` bash
# details
name : Sagar Jagga
roll_no : da25m526
```
# 🎯 Overview

In this assignment, you will take a PyTorch image classification model through the full deployment lifecycle: from local development with proper Git workflows, to containerized training with Docker, to orchestrated deployment on Kubernetes. By the end, you will have a production-style ML pipeline that can train and serve predictions at scale.

# 🎓 Learning Objectives

  1. By completing this assignment, you will be able to:
  2. Structure an ML project repository with proper Git practices (branching, PRs, .gitignore secrets management)
  3. Write multi-stage Dockerfiles optimized for ML workloads
  4. Deploy PyTorch training jobs on Kubernetes using Jobs and persistent storage
  5. Serve a trained model via a Kubernetes Deployment with health checks
  6. Use ConfigMaps and Secrets for environment-specific configuration.
     
# 📋 Prerequisites

 - Python 3.10+, PyTorch experience
 - Docker Desktop installed (or access to a Docker-enabled VM)
 - kubectl CLI installed
 - A Kubernetes cluster (Minikube, kind, or a cloud-managed cluster)
 - A GitHub account.
   
# Part A: Repository Setup

# 🛠️ Repository Architecture
```text
mlops-pytorch-pipeline/
├── README.md
├── .gitignore
├── .dockerignore
├── .gitattributes
├── .github/
│   └── workflows/
│       └── ci.yml
├── src/
│   ├── train.py
│   ├── model.py
│   ├── dataset.py
│   └── serve.py
├── configs/
│   └── training_config.yaml
├── docker/
│   ├── Dockerfile.train
│   └── Dockerfile.serve
├── k8s/
│   ├── namespace.yaml
│   ├── training-job.yaml
│   ├── serving-deployment.yaml
│   ├── serving-service.yaml
│   ├── configmap.yaml
│   └── hpa.yaml
├── requirements/
│   ├── train.txt
│   └── serve.txt
└── tests/
│     └── test_model.py
├── utility/
│   └── setup.py
```
# System & Pipeline Architecture
```plaintext
                         ┌──────────────────────┐
                         │      Developer       │
                         │                      │
                         │  Code / train.py     │
                         │  serve.py / configs  │
                         └──────────┬───────────┘
                                    │
                              Docker Build
                                    │
                    ┌───────────────┴────────────────┐
                    │                                │
                    ▼                                ▼
          ┌──────────────────┐            ┌──────────────────┐
          │ Docker Image     │            │ Docker Image     │
          │ mlops-train:v1   │            │ mlops-serve:v1   │
          │                  │            │                  │
          │ Training         │            │ FastAPI Serving  │
          └────────┬─────────┘            └────────┬─────────┘
                   │                               │
                   │                               │
                   └───────────────┬───────────────┘
                                   │
                                   │ Deploy
                                   ▼
        ┌─────────────────────────────────────────────────────┐
        │                 Kubernetes / Minikube                │
        │                                                     │
        │   ┌──────────────────┐                              │
        │   │    ConfigMap     │                              │
        │   │ training-config  │                              │
        │   └────────┬─────────┘                              │
        │            │                                        │
        │            ▼                                        │
        │   ┌──────────────────┐                              │
        │   │  Training Job    │                              │
        │   │                  │                              │
        │   │ mlops-train:v1   │                              │
        │   └────────┬─────────┘                              │
        │            │                                        │
        │       Train Model                                   │
        │            │                                        │
        │            ▼                                        │
        │   ┌──────────────────┐                              │
        │   │       PVC        │                              │
        │   │    mlops-pvc     │                              │
        │   │                  │                              │
        │   │ best_model.pt    │                              │
        │   └────────┬─────────┘                              │
        │            │                                        │
        │            │ Load Model                             │
        │            ▼                                        │
        │   ┌─────────────────────────┐                       │
        │   │ Serving Deployment      │                       │
        │   │                         │                       │
        │   │ mlops-serve:v1          │                       │
        │   │                         │                       │
        │   │ FastAPI                 │                       │
        │   │ /predict                │                       │
        │   └────────────┬────────────┘                       │
        │                │                                    │
        │                ▼                                    │
        │       ┌──────────────────┐                          │
        │       │ Serving Service  │                          │
        │       │  NodePort / etc. │                          │
        │       └────────┬─────────┘                          │
        │                │                                    │
        │       ┌────────┴─────────┐                          │
        │       │       HPA        │                          │
        │       │ Auto Scaling     │                          │
        │       └──────────────────┘                          │
        │                                                     │
        │                                                     |
        │                                                     |
        │                                                     |
        └──────────────────────┬──────────────────────────────┘
                               │
                               │ HTTP / FastAPI
                               ▼
                       ┌──────────────────┐
                       │  Client / User   │
                       │                  │
                       │ Prediction       │
                       └──────────────────┘
```    
# Model Architecture
```plaintext
model : ResNet-18
input : 3x32x32 pixels
output class :{"airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck"}
Loss & Optimizer: Cross-Entropy Loss with Adam optimizer (lr = 0.001)
```
# 🚀 Quick Start
```bash
# Set up Python virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install training requirements
pip install -r requirements/train.txt

# Install serving requirements
pip install -r requirements/serve.txt
```
# Part B : PyTorch Model
```plaintext
So repo steup , branch creation etc..till that it was PART A. Now will be doing the Part B pytorch model training on `CIFAR-10` dataset.
```
```bash
# 1. train the model locally.
python src/train.py
# output will look like below.
{"epoch": 1, "train_loss": 1.1102, "train_accuracy": 0.6205, "val_loss": 0.8658, "val_accuracy": 0.7079}
{"event": "checkpoint_saved", "path": "checkpoints/classifier_v1.pt"}
# 2. start the fastapi server
uvicorn src.serve:app --host 0.0.0.0 --port 8080 --reload
# 3. check the health endpoint
curl http://localhost:8080/health
# output will come like this
{"status":"healthy","model_loaded":true}
# 4. check the prediction endpoint , please keep the test_image.jpeg in project directory to test from local.
curl -X POST http://localhost:8080/predict -F "image=@test_image.jpeg"
# output will come like  this 
{"filename":"test_image.jpeg","prediction":"horse","confidence":1.0,"probabilities":{"airplane":0.0,"automobile":0.0,"bird":0.0,"cat":0.0,"deer":0.0,"dog":0.0,"frog":1.0,"horse":0.0,"ship":0.0,"truck":0.0}}
```
# Part C : Docker Containerization
```plaintext
now Part B tested successfully , let's do the Part C. 
```
```bash
# verify docker daemon , docker is running fine :
docker info
# 1.Build and test the training image:
# Build image
docker build -f docker/Dockerfile.train -t mlops-train:v1 .
# Run training with mounted volumes
docker run --rm \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/checkpoints:/app/checkpoints \
  mlops-train:v1

#2.build and test the serving image
# Build image
docker build -f docker/Dockerfile.serve -t mlops-serve:v1 .

# Run serving container
docker run -d --rm -p 8080:8080 \
  --name serve-container \
  -v $(pwd)/checkpoints:/app/checkpoints \
  mlops-serve:v1

#3.Test prediction endpoint
# Check health
curl http://localhost:8080/health
# Send test image for prediction
curl -X POST http://localhost:8080/predict -F "image=@test_image.jpeg"

```
# Part D: Kubernetes Training Job + Part E: Kubernetes Model Serving + Part F: End-to-End Validation
```plaintext
now Part C tested successfully , let's do the Part D , E, F for k8s implementation.
```
```bash 
# 1. Start Minikube , don't run in python env/project path , run in nomral WSL , once minikube started execute the command from step 2 from  python env/project path  becuase it need Dockerfile.train and Dockerfile.serve to rebuild the images.
minikube start

# 2. Point local Docker CLI to Minikube's Docker daemon
eval $(minikube -p minikube docker-env)

# 3. Rebuild your Docker images inside Minikube's Docker daemon
docker build -f docker/Dockerfile.train -t mlops-train:v1 .
docker build -f docker/Dockerfile.serve -t mlops-serve:v1 .
```
```bash 
# Demonstrate the full workflow running on your Kubernetes cluster:
# 1 Apply all manifests:
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/training-job.yaml

# Wait for Training to Finish...
# Watch job progress until COMPLETIONS shows 1/1
kubectl get job mlops-train-job -n ml-training -w
# (Press Ctrl+C once COMPLETIONS shows 1/1).
# Check Live Training Logs:
kubectl logs -f job/mlops-train-job -n ml-training
# Check Pod Health & Events:
kubectl get pods -n ml-training
# once succesfully finishes then move to setp 2 below.

# 2 Once training completes, deploy the serving layer:
kubectl apply -f k8s/serving-deployment.yaml
kubectl apply -f k8s/serving-service.yaml
kubectl apply -f k8s/hpa.yaml

# 3 Verify pods are running and healthy:
kubectl get pods -n ml-training
kubectl describe deployment model-serving -n ml-training

# 4 Test the prediction endpoint:
# Port-forward for local testing
kubectl port-forward svc/model-serving 8080:80 -n ml-training
# Send a prediction request
curl -X POST http://localhost:8080/predict -F "image=@test_image.jpeg"

# 5. if issue come port 8080 is used and  Unable to listen on port 8080 then try below one

kubectl port-forward svc/model-serving 8081:80 -n ml-training
# Send a prediction request
curl -X POST http://localhost:8081/predict -F "image=@test_image.jpeg"
```
