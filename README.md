# Safe-Lens — LLM Visualization Platform

An interactive platform for visualizing Large Language Model inference in real time, with token-level probability analysis and automated content moderation via SafeNudge.

**Technologies**: FastAPI · PyTorch · Transformers (Hugging Face) · React · D3.js · Vite · Tailwind CSS · Docker · Kubernetes

---

## Overview

Safe-Lens streams LLM output token by token and exposes the model's internal probability distribution at each step. Users can inspect alternatives, regenerate from any point in the sequence, and optionally activate SafeNudge — a learned guardrail that detects and redirects unsafe generation in real time.

**Key features**:

- **Real-time token streaming**: token-by-token generation with per-token probability distributions
- **Interactive editing**: click any generated token to view its top-k alternatives; select one to regenerate from that point
- **Uncertainty visualization**: token background colour encodes model confidence
- **SafeNudge guardrail**: MLP classifier on hidden states detects unsafe continuations and injects a redirection prompt
- **Prefix support**: optionally seed the assistant turn with a target prefix before generation begins
- **Prometheus metrics**: `/metrics` endpoint for production monitoring

---

## Architecture

### Production (`k8s/`)

Two independent Kubernetes deployments behind an HAProxy ingress with SSL termination:

| Component | Image | Notes |
|-----------|-------|-------|
| Backend | `ghcr.io/dataresponsibly/safe-lens-backend:latest` | Requires 1 GPU |
| Frontend | `ghcr.io/dataresponsibly/safe-lens-frontend:latest` | Lightweight, no GPU |

GPU affinity targets: Quadro RTX 8000, NVIDIA A100 (40 GB or 80 GB PCIe), or NVIDIA RTX A6000.

Backend resource allocation:
- Requests: 4 CPU / 16 Gi RAM
- Limits: 8 CPU / 24 Gi RAM / 1 GPU

The startup probe allows up to 20 minutes for model loading before liveness/readiness checks begin.

---

## Getting Started

### Prerequisites

- **Hugging Face token** with access to [`meta-llama/Llama-3.2-1B-Instruct`](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct)
- **kubectl** configured against the target cluster

### Deploy

```bash
# Create the HuggingFace token secret (once)
kubectl create secret generic hf-token --from-literal=token=YOUR_TOKEN

# Deploy all services (backend, frontend, ingress, monitoring)
kubectl apply -f k8s/

# Check status
kubectl get pods
kubectl get svc
```

### Local Access (port-forward)

```bash
# Forward the frontend to http://localhost:8080
kubectl port-forward deployment/frontend 8080:80

# Or forward the backend API directly to http://localhost:8000
kubectl port-forward deployment/backend 8000:8000
```

### Making the App Public

The ingress in `k8s/ingress.yaml` controls public visibility via the `hpc.nyu.edu/access` annotation:

```yaml
# Public (accessible outside the cluster)
annotations:
  hpc.nyu.edu/access: "public"

# Private (remove the annotation or set to "private")
annotations: {}
```

After editing the annotation, apply with `replace` (not `apply`) to force the ingress controller to pick up the change:

```bash
kubectl replace -f k8s/ingress.yaml
```

**Production URL**: `https://safe-lens.users.hsrn.nyu.edu`

---

## API

All endpoints accept `application/x-www-form-urlencoded` (form-encoded) POST bodies. Responses are newline-delimited JSON.

### `GET /`

Health check. Returns a list of available API routes.

### `POST /generate`

Stream a token-by-token generation response.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `init_prompt` | string | required | User prompt (1–10 000 chars) |
| `k` | int | 30 | Top-k sampling breadth (1–50) |
| `T` | float | 1.0 | Sampling temperature (>0, ≤10) |
| `tau` | float | 0.5 | SafeNudge intervention threshold (0–1) |
| `max_new_tokens` | int | 100 | Max tokens to generate (1–4096) |
| `sleep_time` | float | 0.0 | Per-token delay in seconds (0–2) |
| `random_state` | int | null | RNG seed for reproducibility |
| `safenudge` | bool | false | Enable SafeNudge moderation |
| `target` | string | null | Assistant turn prefix |

Each streamed object:

```json
{
  "idx_counter": 4,
  "texts": ["Hello", " Hi", " Hey", ...],
  "token_ids": [9906, 13347, 19182, ...],
  "probs": [0.42, 0.31, 0.12, ...],
  "selected_idx": 0,
  "selected_text": "Hello"
}
```

When SafeNudge fires, a sentinel object is emitted before generation resumes:

```json
{ "idx_counter": -1, "texts": [], "token_ids": [], "probs": [], "selected_idx": -1, "selected_text": "[NUDGE]" }
```

### `POST /regenerate`

Regenerate from a chosen alternative token at a given position.

| Parameter | Type | Description |
|-----------|------|-------------|
| `init_prompt` | string | Original prompt |
| `content` | string | JSON-serialised token history from the client |
| `token_pos` | int | Index of the token to branch from |
| `new_token` | string | Alternative token text to substitute |
| `k`, `T`, `max_new_tokens`, `sleep_time`, `random_state` | — | Same as `/generate` |

### `GET /metrics`

Prometheus-compatible metrics endpoint.

---

## SafeNudge

SafeNudge is the cognitive guardrail that moderates unsafe LLM outputs without blocking generation entirely.

**How it works**:

1. Generate tokens normally for the first 4 positions (warmup).
2. From position 5 onward, extract the last hidden state from the model at each step.
3. Run the hidden state through a pre-trained MLP classifier (`api/artifacts/clf_mlp_llama_1b.pkl`).
4. If `P(unsafe) ≥ tau`, append a redirection prompt to the full context (including the unsafe tokens generated so far) and continue:

> *"Sorry, I was going to generate an unsafe response. Instead, let me correct that and make sure the response is very safe, ethical and cannot be used to harm society. Here is an alternate, safe answer to your question:"*

5. Generation continues from the redirected context. No tokens are discarded.

**Implementation notes**:

- The classifier uses scikit-learn's `predict_proba` on a flattened hidden-state vector.
- `past_key_values` is reset after a nudge to force a full KV-cache reprefill with the new context.
- WildGuard and Qwen3Guard integrations exist in `api/` but are disabled in production.

---

## Code Quality

```bash
# Format with Black
make code-format

# Lint (black check, mypy, flake8, pylint)
make code-analysis

# Clean build artifacts and caches
make clean
```

---

## Testing

Playwright-based stress test:

```bash
pip install playwright
playwright install chromium

# Full stress test against the deployed instance
python tests/stress_test_playwright.py \
  --url "https://safe-lens.users.hsrn.nyu.edu/" \
  --users 20 \
  --iterations 5 \
  --timeout-ms 120000

# Quick sanity check
python tests/stress_test_playwright.py --users 5 --iterations 2
```

---

## Project Structure

```
safe-lens/
├── api/                            # FastAPI backend
│   ├── api.py                      # Routes: /generate, /regenerate, /metrics
│   ├── safenudge.py                # ModelWrapper + SafeNudge guardrail
│   ├── wildguard_safenudge.py      # WildGuard integration (disabled)
│   ├── qwenqguard_safenudge.py     # Qwen3Guard integration (disabled)
│   ├── _loader.py                  # Model loading (AutoModelForCausalLM)
│   ├── _output_handler.py          # Token streaming and CUDA memory management
│   ├── metrics.py                  # Prometheus metrics router
│   └── artifacts/
│       └── clf_mlp_llama_1b.pkl    # Pre-trained SafeNudge MLP classifier
├── client/                         # React frontend (Vite + Tailwind + D3.js)
│   ├── src/
│   │   ├── App.jsx                 # Application state and layout
│   │   ├── components/
│   │   │   ├── ChatInput.jsx
│   │   │   ├── ChatOutput.jsx      # Token rendering with uncertainty colours
│   │   │   ├── Barplot.jsx         # D3.js probability bar chart
│   │   │   └── SettingsPanel.jsx
│   │   └── api/stream.js           # Streaming fetch helpers
│   └── index.html
├── dev/                            # Local development stack
│   ├── docker-compose.yml
│   ├── Dockerfile.backend          # CPU-only PyTorch image
│   ├── Dockerfile.frontend         # Node/Vite image
│   └── nginx.conf
├── k8s/                            # Kubernetes manifests
│   ├── backend.yaml                # GPU deployment + ClusterIP service
│   ├── frontend.yaml               # Frontend deployment + service
│   ├── ingress.yaml                # HAProxy ingress with SSL
│   ├── monitoring-prometheus-*.yaml
│   └── monitoring-grafana.yaml
├── examples/                       # Standalone usage scripts
├── tests/
│   └── stress_test_playwright.py
├── Dockerfile.backend              # Production GPU image
├── Dockerfile.frontend             # Production frontend image
├── Makefile
├── requirements.txt
└── tox.ini
```

---

## Environment

**Development** — create `api/.env`:

```
HF_TOKEN=your_token_here
```

**Production** — Kubernetes secret:

```bash
kubectl create secret generic hf-token --from-literal=token=YOUR_TOKEN
```

PyTorch is installed in production via the CUDA 12.1 index:

```
--index-url https://download.pytorch.org/whl/cu121
```
