#!/bin/bash
# Download inference server files from GitHub and start the server
# This is cleaner than base64 encoding

set -e
mkdir -p /workspace/inference_server
cd /workspace/inference_server

# Configuration - Set these environment variables or defaults
GITHUB_USER="${GITHUB_USER:-}"
GIST_ID="${GIST_ID:-}"
GITHUB_REPO="${GITHUB_REPO:-}"  # Format: "username/repo"
BRANCH="${BRANCH:-main}"

# Install curl if not available
if ! command -v curl &> /dev/null; then
    apt-get update && apt-get install -y curl -q
fi

echo "📥 Downloading inference server files..."

# Determine download method
if [ -n "$GIST_ID" ] && [ -n "$GITHUB_USER" ]; then
    # Use GitHub Gist
    BASE_URL="https://gist.githubusercontent.com/${GITHUB_USER}/${GIST_ID}/raw"
    echo "Using GitHub Gist: ${BASE_URL}"
elif [ -n "$GITHUB_REPO" ]; then
    # Use GitHub repo
    BASE_URL="https://raw.githubusercontent.com/${GITHUB_REPO}/${BRANCH}/inference_server"
    echo "Using GitHub Repo: ${BASE_URL}"
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

# Download files
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

