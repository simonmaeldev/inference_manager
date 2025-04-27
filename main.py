import logging
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from src.routes.mcp import setup_mcp_server
import uvicorn
import time
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="LocalAI Inference Manager",
    description="Queue-based inference server for LocalAI",
    version="0.1.0"
)

async def log_request(request: Request, call_next):
    """Middleware to log incoming requests and responses"""
    start_time = time.time()
    
    # Log request details
    logger.info(f"Request: {request.method} {request.url.path}")
    
    try:
        response = await call_next(request)
    except Exception as e:
        logger.error(f"Request failed: {str(e)}")
        raise
    
    # Calculate processing time
    process_time = (time.time() - start_time) * 1000
    formatted_time = f"{process_time:.2f}ms"
    
    # Log response details
    logger.info(
        f"Response: {request.method} {request.url.path} "
        f"Status: {response.status_code} Time: {formatted_time}"
    )
    
    return response

# Add request logging middleware
app.middleware('http')(log_request)

# Setup CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Setup MCP server
mcp_server = setup_mcp_server()
app.mount("/mcp", mcp_server.app)

@app.get("/health")
async def health_check():
    """Basic health check endpoint"""
    return {"status": "healthy", "version": app.version}

@app.get("/queue-status")
async def queue_status():
    """Get current queue status and system metrics"""
    from src.routes.mcp import queues
    import psutil
    
    # Get queue metrics
    queue_metrics = {
        model: {
            "pending_requests": len(queue),
            "priority": queues.priority_order.index(next(
                t for t, models in queues.model_sets.items()
                if model in models
            )) + 1  # 1-based priority
        }
        for model, queue in queues.queues.items()
    }
    
    # Get system metrics
    system_metrics = {
        "cpu_percent": psutil.cpu_percent(),
        "memory_percent": psutil.virtual_memory().percent,
        "timestamp": datetime.now().isoformat()
    }
    
    return {
        "queues": queue_metrics,
        "system": system_metrics
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
