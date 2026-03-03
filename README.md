# Luxembourg - LLM Visualization Platform

A unified platform for visualizing and interacting with Large Language Models, featuring real-time token analysis, cognitive guardrails, and advanced safety mechanisms in a single integrated container.


## Overview

This project provides an interactive visualization platform for Large Language Models that combines:

- **Integrated Backend API & Frontend**: Combined FastAPI service and web interface in a single container
- **Real-time Token Analysis**: Interactive visualization of model predictions and probabilities  
- **Safety Features**: Integration with WildGuard SafeNudge for content moderation
- **Single Container Architecture**: Simplified deployment with nginx proxy and FastAPI backend

## Architecture

The platform uses a **single-container architecture** combining both frontend and backend:

### Unified Container (llm-viz-combined)
- **Backend**: Python FastAPI (port 8000)
- **Frontend**: HTML/CSS/JavaScript with D3.js visualizations
- **Proxy**: Nginx serving static files and proxying API calls (port 80)
- **Features**:
  - LLM model serving (Llama-3.2-1B-Instruct)
  - Token probability analysis with interactive charts
  - Safety content filtering with WildGuard
  - Response modification and regeneration
  - RESTful API endpoints with form-based data submission

## Key Features

- 🔍 **Token Analysis**: Real-time visualization of token probabilities and model predictions
- 🛡️ **Safety Guardrails**: Automated content moderation using WildGuard safety models
- 📊 **Interactive Visualizations**: Dynamic D3.js charts for model behavior analysis
- 🎯 **Token Regeneration**: Click-to-regenerate functionality for exploring alternative outputs
- 🔄 **Streaming Responses**: Real-time token generation with progress visualization

## Installation & Deployment

### Prerequisites

- **Hugging Face Token**: Required for model access ([get token here](https://huggingface.co/settings/tokens))
- **Model Access**: Request permission for [Llama-3.2-1B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct)
- **Container Runtime**: Docker or Podman for local development
- **Kubernetes Cluster**: For production deployment with GPU support

### Local Development

```bash
# Clone the repository
git clone https://github.com/DataResponsibly/safe-lens.git

# Change to the project directory
cd safe-lens

# Build and run the integrated container
make run

# Access the application
open http://localhost:8080
```

### Kubernetes Deployment

For production deployment on Kubernetes clusters:

```bash
# Create the HuggingFace token secret
kubectl create secret generic hf-token --from-literal=token=your_hf_token_here

# Deploy the application
kubectl apply -f kubernetes.yaml

# Check deployment status
kubectl get pods
kubectl get svc

# Port forward for local testing
kubectl port-forward deployment/llm-viz 8080:80
```

The Kubernetes deployment features:
- **Single Container Pod**: Simplified architecture with integrated frontend/backend
- **Resource Management**: GPU allocation for ML workloads, optimized memory usage  
- **Load Balancing**: HAProxy ingress with SSL termination
- **Rolling Updates**: Zero-downtime deployments
- **Auto-scaling**: Configurable based on CPU/memory usage

### Access Points

- **Local Development**: 
  - Web Interface: `http://localhost:8080`
  - API Documentation: `http://localhost:8080/docs` (when running FastAPI in debug mode)

- **Production**: 
  - Web Interface: `https://llm-viz.users.hsrn.nyu.edu`

## API Documentation

### Core Endpoints

- `POST /generate` - Generate text responses with token analysis (accepts form data)
- `POST /regenerate` - Regenerate text from a specific token position (accepts form data)
- `GET /` - Health check and basic information

### Request Format

Both endpoints now accept form-encoded data instead of query parameters to avoid URL length limitations:

```javascript
// Example generate request
const requestData = {
    init_prompt: "Your prompt here",
    safenudge: false,
    k: 20,
    T: 1.3,
    max_new_tokens: 300,
    verbose: false,
    random_state: 12345,
    sleep_time: 0.1
};

fetch('/generate', {
    method: 'POST',
    body: new URLSearchParams(requestData)
});
```

## Project Structure

```
luxembourg/
├── api/                        # Backend FastAPI application
│   ├── api.py                  # Main API endpoints with form data handling
│   ├── safenudge.py            # Cognitive bias detection
│   ├── wildguard_safenudge.py  # WildGuard content safety filtering
│   ├── _loader.py              # Model loading utilities
│   ├── _output_handler.py      # Response processing utilities
│   └── artifacts/              # Pre-trained model artifacts
├── client/                     # Frontend web application
│   ├── main.html               # Main application interface
│   ├── css/                    # Stylesheets
│   └── js/                     # JavaScript with D3.js visualizations
├── examples/                   # Usage examples and demos
├── kubernetes.yaml             # Kubernetes deployment configuration
├── Dockerfile                  # Unified container build instructions
├── nginx.conf                  # Nginx configuration for proxying
├── start.sh                    # Container startup script
└── requirements.txt            # Python dependencies
```

## How It Works

1. **User Interface**: Users interact with a clean web interface to input prompts
2. **Token Generation**: The system generates text token by token, showing probabilities for each
3. **Visualization**: Real-time charts display token alternatives and their probabilities
4. **Safety Filtering**: WildGuard SafeNudge monitors content for safety issues
5. **Interactive Editing**: Users can click on alternative tokens to regenerate from that point
6. **Streaming Response**: All generation happens with real-time streaming for immediate feedback 