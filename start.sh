#!/bin/bash
# Startup script for PyTorch inference server
# This script installs dependencies and starts the FastAPI server

set -e

echo "🚀 Starting PyTorch Inference Server Setup..."

# Check if Python 3 is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Installing Python 3..."
    apt-get update && apt-get install -y python3 python3-pip
fi

# Navigate to inference server directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Install dependencies
echo "📦 Installing Python dependencies..."
pip3 install --upgrade pip
pip3 install -r requirements.txt

# Set environment variables with defaults
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
echo "   Host: $SERVER_HOST"
echo "   Device: $TORCH_DEVICE"
echo "   Max Concurrent Requests: $MAX_CONCURRENT_REQUESTS"
echo "   Load in 8-bit: $MODEL_LOAD_IN_8BIT"
echo ""

# Start the server
echo "🎯 Starting FastAPI server..."
exec python3 server.py
