#!/usr/bin/env python3
"""
FastAPI server with Ollama-compatible API endpoints for PyTorch inference.
Supports vision models like Qwen2-VL/Qwen2.5-VL with parallel processing.
"""

import os
import base64
import json
import asyncio
import uuid
import threading
from typing import Optional, List, Dict, Any, AsyncGenerator
from io import BytesIO
from dataclasses import dataclass
from datetime import datetime

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
from PIL import Image
import uvicorn
import torch
from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoProcessor,
    Qwen2VLProcessor,
    TextIteratorStreamer,
)

app = FastAPI(title="PyTorch Inference Server")

# Global model instances
model: Optional[torch.nn.Module] = None
tokenizer: Optional[Any] = None
processor: Optional[Any] = None
model_name: str = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-VL-7B-Instruct")
port: int = int(os.getenv("SERVER_PORT", "11434"))
host: str = os.getenv("SERVER_HOST", "0.0.0.0")
device: str = os.getenv("TORCH_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
max_concurrent_requests: int = int(os.getenv("MAX_CONCURRENT_REQUESTS", "4"))
load_in_8bit: bool = os.getenv("MODEL_LOAD_IN_8BIT", "false").lower() == "true"

# Request queue and worker pool
request_queue: asyncio.Queue = asyncio.Queue()
workers: List[asyncio.Task] = []


@dataclass
class RequestItem:
    """Container for request data and response callbacks."""
    request_id: str
    model: str
    prompt: str
    images: Optional[List[str]]
    stream: bool
    temperature: float
    top_p: float
    max_tokens: int
    response_queue: asyncio.Queue
    error: Optional[Exception] = None


class GenerateRequest(BaseModel):
    model: str
    prompt: str
    images: Optional[List[str]] = None  # Base64 encoded images
    stream: bool = False
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    max_tokens: Optional[int] = None


class ModelInfo(BaseModel):
    name: str
    modified_at: str
    size: int
    digest: str
    details: Dict[str, Any]


def decode_base64_image(image_b64: str) -> Image.Image:
    """Decode base64 image string to PIL Image."""
    try:
        # Remove data URL prefix if present
        if "," in image_b64:
            image_b64 = image_b64.split(",", 1)[1]
        
        image_data = base64.b64decode(image_b64)
        image = Image.open(BytesIO(image_data))
        return image
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image data: {str(e)}")


def initialize_model():
    """Initialize the PyTorch model and tokenizer/processor."""
    global model, tokenizer, processor
    
    if model is not None:
        return
    
    print(f"Initializing PyTorch model: {model_name}")
    print(f"Device: {device}")
    print(f"Load in 8-bit: {load_in_8bit}")
    
    try:
        # Check if this is a vision model
        is_vision_model = "VL" in model_name or "vision" in model_name.lower()
        
        if is_vision_model:
            # For vision models, use processor
            print("Loading vision model processor...")
            try:
                processor = AutoProcessor.from_pretrained(
                    model_name,
                    trust_remote_code=True,
                    cache_dir=os.getenv("MODEL_CACHE_DIR", "/workspace/models"),
                )
                tokenizer = processor.tokenizer
            except Exception:
                # Fallback to Qwen2VLProcessor if AutoProcessor fails
                processor = Qwen2VLProcessor.from_pretrained(
                    model_name,
                    trust_remote_code=True,
                    cache_dir=os.getenv("MODEL_CACHE_DIR", "/workspace/models"),
                )
                tokenizer = processor.tokenizer
        else:
            # For text-only models, use tokenizer
            print("Loading tokenizer...")
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True,
                cache_dir=os.getenv("MODEL_CACHE_DIR", "/workspace/models"),
            )
        
        # Load model
        print("Loading model...")
        model_kwargs = {
            "trust_remote_code": True,
            "cache_dir": os.getenv("MODEL_CACHE_DIR", "/workspace/models"),
            "torch_dtype": torch.float16 if device == "cuda" else torch.float32,
        }
        
        if load_in_8bit and device == "cuda":
            try:
                from transformers import BitsAndBytesConfig
                quantization_config = BitsAndBytesConfig(load_in_8bit=True)
                model_kwargs["quantization_config"] = quantization_config
            except ImportError:
                print("Warning: bitsandbytes not available, loading in full precision")
        
        # Use AutoModel for vision-language models, AutoModelForCausalLM for text-only models
        if is_vision_model:
            print("Using AutoModel for vision-language model...")
            model = AutoModel.from_pretrained(
                model_name,
                **model_kwargs
            )
        else:
            print("Using AutoModelForCausalLM for text-only model...")
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                **model_kwargs
            )
        
        # Move model to device
        if device == "cuda":
            model = model.to(device)
            if hasattr(model, "half"):
                model = model.half()
        
        model.eval()
        
        print(f"✅ PyTorch model initialized successfully")
        print(f"   Model device: {next(model.parameters()).device}")
        print(f"   Model dtype: {next(model.parameters()).dtype}")
        
    except Exception as e:
        print(f"❌ Failed to initialize PyTorch model: {str(e)}")
        raise


async def process_request(request_item: RequestItem):
    """Process a single inference request."""
    try:
        # Prepare inputs
        if request_item.images and len(request_item.images) > 0:
            # Vision model processing
            images = [decode_base64_image(img) for img in request_item.images]
            
            if processor is None:
                raise HTTPException(status_code=500, detail="Processor not initialized")
            
            # Prepare inputs with images
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": img} for img in images
                    ] + [{"type": "text", "text": request_item.prompt}]
                }
            ]
            
            # Process inputs
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = processor.process_vision_info(messages)
            
            inputs = processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt"
            )
            
            inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
            
        else:
            # Text-only processing
            if tokenizer is None:
                raise HTTPException(status_code=500, detail="Tokenizer not initialized")
            
            inputs = tokenizer(
                request_item.prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=int(os.getenv("MAX_MODEL_LEN", "4096")),
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Generation parameters
        generation_config = {
            "max_new_tokens": request_item.max_tokens,
            "temperature": request_item.temperature,
            "top_p": request_item.top_p,
            "do_sample": request_item.temperature > 0,
        }
        
        # Generate response
        if request_item.stream:
            # Streaming generation using TextIteratorStreamer
            async def generate_stream():
                try:
                    # Get the appropriate tokenizer for streamer
                    streamer_tokenizer = tokenizer if tokenizer else (processor.tokenizer if processor else None)
                    if streamer_tokenizer is None:
                        raise ValueError("No tokenizer available for streaming")
                    
                    # Create streamer
                    streamer = TextIteratorStreamer(
                        streamer_tokenizer,
                        skip_prompt=True,
                        skip_special_tokens=True,
                    )
                    
                    # Prepare generation kwargs
                    generation_kwargs = {
                        **inputs,
                        **generation_config,
                        "streamer": streamer,
                    }
                    
                    # Add pad_token_id and eos_token_id if available
                    if tokenizer:
                        if tokenizer.pad_token_id is not None:
                            generation_kwargs["pad_token_id"] = tokenizer.pad_token_id
                        if tokenizer.eos_token_id is not None:
                            generation_kwargs["eos_token_id"] = tokenizer.eos_token_id
                    elif processor and processor.tokenizer:
                        if processor.tokenizer.pad_token_id is not None:
                            generation_kwargs["pad_token_id"] = processor.tokenizer.pad_token_id
                        if processor.tokenizer.eos_token_id is not None:
                            generation_kwargs["eos_token_id"] = processor.tokenizer.eos_token_id
                    
                    # Run generation in a separate thread
                    def run_generation():
                        with torch.no_grad():
                            model.generate(**generation_kwargs)
                    
                    generation_thread = threading.Thread(target=run_generation)
                    generation_thread.start()
                    
                    # Stream tokens as they're generated
                    for text in streamer:
                        await request_item.response_queue.put({
                            "response": text,
                            "done": False
                        })
                    
                    # Wait for generation thread to finish
                    generation_thread.join()
                    
                    # Send final chunk
                    await request_item.response_queue.put({
                        "response": "",
                        "done": True
                    })
                except Exception as e:
                    await request_item.response_queue.put({
                        "error": str(e),
                        "done": True
                    })
            
            await generate_stream()
        else:
            # Non-streaming generation
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    **generation_config,
                    pad_token_id=tokenizer.pad_token_id if tokenizer else None,
                    eos_token_id=tokenizer.eos_token_id if tokenizer else None,
                )
            
            # Decode response
            if processor:
                generated_text = processor.decode(outputs[0], skip_special_tokens=True)
                # Remove prompt from response
                if request_item.prompt in generated_text:
                    generated_text = generated_text.replace(request_item.prompt, "").strip()
            else:
                input_length = inputs["input_ids"].shape[1]
                generated_text = tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)
            
            # Send result
            await request_item.response_queue.put({
                "response": generated_text,
                "done": True
            })
        
        # Clean up GPU memory
        if device == "cuda":
            torch.cuda.empty_cache()
            
    except Exception as e:
        request_item.error = e
        await request_item.response_queue.put({
            "error": str(e),
            "done": True
        })


async def worker(worker_id: int):
    """Worker coroutine that processes requests from the queue."""
    print(f"Worker {worker_id} started")
    while True:
        try:
            request_item = await request_queue.get()
            if request_item is None:  # Shutdown signal
                break
            
            await process_request(request_item)
            request_queue.task_done()
        except Exception as e:
            print(f"Worker {worker_id} error: {str(e)}")
            if request_item:
                request_item.error = e
                await request_item.response_queue.put({
                    "error": str(e),
                    "done": True
                })


async def start_workers():
    """Start worker pool for parallel request processing."""
    global workers
    workers = [asyncio.create_task(worker(i)) for i in range(max_concurrent_requests)]
    print(f"✅ Started {max_concurrent_requests} worker(s) for parallel processing")


async def stop_workers():
    """Stop all workers."""
    global workers
    # Send shutdown signal to all workers
    for _ in workers:
        await request_queue.put(None)
    # Wait for workers to finish
    await asyncio.gather(*workers, return_exceptions=True)
    workers = []


@app.on_event("startup")
async def startup_event():
    """Initialize model and workers on startup."""
    # Run model initialization in executor to avoid blocking
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, initialize_model)
    await start_workers()


@app.on_event("shutdown")
async def shutdown_event():
    """Clean up workers on shutdown."""
    await stop_workers()


@app.get("/api/tags")
async def list_models():
    """
    Ollama-compatible endpoint to list loaded models.
    Returns model information for health checks.
    """
    return {
        "models": [
            {
                "name": model_name,
                "modified_at": datetime.now().isoformat() + "Z",
                "size": 0,  # Size not available without model info
                "digest": "",
                "details": {
                    "parent_model": "",
                    "format": "pytorch",
                    "family": "qwen" if "qwen" in model_name.lower() else "unknown",
                    "families": ["qwen"] if "qwen" in model_name.lower() else ["unknown"],
                    "parameter_size": "7B" if "7B" in model_name else "unknown",
                    "quantization_level": "8bit" if load_in_8bit else "fp16" if device == "cuda" else "fp32"
                }
            }
        ]
    }


@app.post("/api/generate")
async def generate(request: GenerateRequest):
    """
    Ollama-compatible endpoint for model inference.
    Supports vision models with base64-encoded images.
    Handles concurrent requests via worker pool.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not initialized")
    
    if request.model != model_name:
        raise HTTPException(
            status_code=400,
            detail=f"Model '{request.model}' not available. Only '{model_name}' is loaded."
        )
    
    # Create request item
    request_id = str(uuid.uuid4())
    response_queue = asyncio.Queue()
    
    request_item = RequestItem(
        request_id=request_id,
        model=request.model,
        prompt=request.prompt,
        images=request.images,
        stream=request.stream,
        temperature=request.temperature if request.temperature is not None else 0.7,
        top_p=request.top_p if request.top_p is not None else 0.9,
        max_tokens=request.max_tokens if request.max_tokens is not None else 512,
        response_queue=response_queue,
    )
    
    # Add to queue
    await request_queue.put(request_item)
    
    try:
        if request.stream:
            # Streaming response
            async def generate_stream() -> AsyncGenerator[str, None]:
                while True:
                    result = await response_queue.get()
                    if result.get("error"):
                        yield f'data: {json.dumps({"error": result["error"], "done": True})}\n\n'
                        break
                    
                    yield f'data: {json.dumps({"response": result.get("response", ""), "done": result.get("done", False)})}\n\n'
                    
                    if result.get("done", False):
                        break
            
            return StreamingResponse(generate_stream(), media_type="text/event-stream")
        else:
            # Non-streaming response - wait for result
            result = await response_queue.get()
            
            if result.get("error"):
                raise HTTPException(status_code=500, detail=result["error"])
            
            generated_text = result.get("response", "")
            
            return {
                "model": request.model,
                "created_at": datetime.now().isoformat() + "Z",
                "response": generated_text,
                "done": True,
                "context": [],
                "total_duration": 0,
                "load_duration": 0,
                "prompt_eval_count": 0,
                "prompt_eval_duration": 0,
                "eval_count": len(generated_text.split()) if generated_text else 0,
                "eval_duration": 0
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation error: {str(e)}")


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model": model_name,
        "model_ready": model is not None,
        "device": device,
        "workers": len(workers),
        "queue_size": request_queue.qsize()
    }


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "PyTorch Inference Server",
        "model": model_name,
        "device": device,
        "max_concurrent_requests": max_concurrent_requests,
        "endpoints": ["/api/generate", "/api/tags", "/health"]
    }


if __name__ == "__main__":
    print(f"Starting PyTorch inference server on {host}:{port}")
    print(f"Model: {model_name}")
    print(f"Max concurrent requests: {max_concurrent_requests}")
    uvicorn.run(app, host=host, port=port)
