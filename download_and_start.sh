#!/bin/bash
# Clone inference server files from GitHub and start the server
# This is cleaner and more reliable than downloading individual files

set -e
mkdir -p /workspace/inference_server
cd /workspace/inference_server

# Configuration - Set these environment variables or defaults
GITHUB_USER="${GITHUB_USER:-}"
GIST_ID="${GIST_ID:-}"
GITHUB_REPO="${GITHUB_REPO:-}"  # Format: "username/repo"
BRANCH="${BRANCH:-main}"
GITHUB_TOKEN="${GITHUB_TOKEN:-}"

# Install git if not available
if ! command -v git &> /dev/null; then
    apt-get update && apt-get install -y git -q
fi

echo "📥 Cloning inference server repository..."

# Determine download method
if [ -n "$GIST_ID" ] && [ -n "$GITHUB_USER" ]; then
    # Use GitHub Gist (still use curl for gists as they don't support git clone easily)
    echo "⚠️  Gist support: Using direct download for Gist..."
    BASE_URL="https://gist.githubusercontent.com/${GITHUB_USER}/${GIST_ID}/raw"
    echo "Using GitHub Gist: ${BASE_URL}"
    
    # Install curl if not available
    if ! command -v curl &> /dev/null; then
        apt-get update && apt-get install -y curl -q
    fi
    
    echo "Downloading server.py..."
    curl -sL "${BASE_URL}/server.py" -o server.py || {
        echo "❌ Failed to download server.py"
        exit 1
    }
    
    echo "Downloading requirements.txt..."
    curl -sL "${BASE_URL}/requirements.txt" -o requirements.txt || {
        echo "⚠️  Failed to download requirements.txt, using defaults..."
        cat > requirements.txt << 'REQEOF'
torch>=2.0.0
transformers>=4.35.0
accelerate>=0.25.0
fastapi>=0.104.0
uvicorn[standard]>=0.24.0
Pillow>=10.0.0
pydantic>=2.0.0
bitsandbytes>=0.41.0
REQEOF
    }
elif [ -n "$GITHUB_REPO" ]; then
    # Clone GitHub repository (preferred method)
    REPO_NAME=$(basename "$GITHUB_REPO")
    CLONE_DIR="/tmp/${REPO_NAME}-${BRANCH}"
    
    # Clean up any existing clone
    rm -rf "$CLONE_DIR"
    
    # Build clone URL (support private repos with token)
    if [ -n "$GITHUB_TOKEN" ]; then
        CLONE_URL="https://${GITHUB_TOKEN}@github.com/${GITHUB_REPO}.git"
        echo "📥 Downloading from GitHub: ${GITHUB_REPO} (${BRANCH})"
        echo "   User: ${GITHUB_USER:-anonymous}"
        echo "   Cloning private repository..."
    else
        CLONE_URL="https://github.com/${GITHUB_REPO}.git"
        echo "📥 Downloading from GitHub: ${GITHUB_REPO} (${BRANCH})"
        echo "   User: ${GITHUB_USER:-anonymous}"
        echo "   Cloning public repository..."
    fi
    
    # Clone the repository
    git clone --depth 1 --branch "$BRANCH" "$CLONE_URL" "$CLONE_DIR" || {
        echo "❌ Failed to clone repository. Check your GITHUB_REPO and BRANCH settings."
        exit 1
    }
    
    # Copy files from cloned repo to workspace
    # Try inference_server directory first, then root
    if [ -f "${CLONE_DIR}/inference_server/server.py" ]; then
        echo "✅ Found files in inference_server/ directory"
        cp "${CLONE_DIR}/inference_server/"*.py . 2>/dev/null || true
        cp "${CLONE_DIR}/inference_server/"requirements.txt . 2>/dev/null || true
    elif [ -f "${CLONE_DIR}/server.py" ]; then
        echo "✅ Found files in repository root"
        cp "${CLONE_DIR}/server.py" . || {
            echo "❌ Failed to copy server.py"
            exit 1
        }
        if [ -f "${CLONE_DIR}/requirements.txt" ]; then
            cp "${CLONE_DIR}/requirements.txt" .
        fi
    else
        echo "❌ Could not find server.py in repository"
        exit 1
    fi
    
    # Clean up cloned repository
    rm -rf "$CLONE_DIR"
    echo "✅ Successfully cloned and copied files"
    
    # Ensure requirements.txt exists
    if [ ! -f "requirements.txt" ]; then
        echo "⚠️  requirements.txt not found, using defaults..."
        cat > requirements.txt << 'REQEOF'
torch>=2.0.0
transformers>=4.35.0
accelerate>=0.25.0
fastapi>=0.104.0
uvicorn[standard]>=0.24.0
Pillow>=10.0.0
pydantic>=2.0.0
bitsandbytes>=0.41.0
REQEOF
    fi
else
    # Fallback: Use base64 (current method)
    echo "⚠️  No GitHub URL provided, using base64 fallback..."
    python3 << 'PYFALLBACK'
import base64
import os
server_b64 = '${SERVER_PY_B64}'
if server_b64 and server_b64 != '${SERVER_PY_B64}':
    with open('/workspace/inference_server/server.py', 'wb') as f:
        f.write(base64.b64decode(server_b64))
PYFALLBACK
    exit 0
fi

# Install dependencies
echo "📦 Installing dependencies..."
pip3 install --upgrade pip -q
pip3 install -r requirements.txt -q

# Set environment variables
export MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-VL-7B-Instruct}"
export SERVER_PORT="${SERVER_PORT:-11434}"
export SERVER_HOST="${SERVER_HOST:-0.0.0.0}"
export TORCH_DEVICE="${TORCH_DEVICE:-cuda}"
export MAX_CONCURRENT_REQUESTS="${MAX_CONCURRENT_REQUESTS:-4}"
export MODEL_LOAD_IN_8BIT="${MODEL_LOAD_IN_8BIT:-false}"
export MODEL_CACHE_DIR="${MODEL_CACHE_DIR:-/workspace/models}"
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"

echo "✅ Configuration:"
echo "   Model: $MODEL_NAME"
echo "   Port: $SERVER_PORT"
echo "   Device: $TORCH_DEVICE"

# Start the server
echo "🚀 Starting PyTorch Inference Server..."
exec python3 server.py

