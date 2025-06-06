
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from mcp.server.fastmcp import FastMCP
import uvicorn

from src.models.schemas import ChatRequest
from src.routes.tools import Tools
from src.utils.ollama_client import get_ollama_tags
import asyncio
from src.models.queues import InferenceQueues
queues = InferenceQueues()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup logic
    asyncio.create_task(queues.process_queues_forever())
    yield
    # Shutdown logic would go here

# Create FastAPI app
app = FastAPI(
    title="LocalAI Inference Manager",
    description="Queue-based inference server for LocalAI",
    version="0.1.0",
    docs_url="/docs",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ollama-compatible endpoints (for N8N Ollama node)
@app.get("/ollama/")
async def ollama_root():
    """Root endpoint that mimics Ollama's default response"""
    print("Ollama root endpoint called")
    return "Ollama is running"

@app.get("/ollama/api/tags")
async def ollama_list_models():
    """List local models by querying Ollama API"""
    print("Endpoint called: /ollama/api/tags")
    return await get_ollama_tags()

@app.post("/ollama/api/chat")
async def ollama_generate_text(request: ChatRequest):
    """Ollama-compatible chat endpoint"""
    print("Endpoint called: /ollama/api/chat")
    return await Tools.generate_text(request.messages, queues, request.model, request.temperature)

@app.post("/ollama/api/embed")
async def ollama_generate_embeddings():
    """Generate embeddings"""
    print("Endpoint called: /ollama/api/embed")
    return {"embeddings": []}

# Health check endpoint
@app.get("/health")
async def health_check():
    """Basic health check endpoint"""
    return {"status": "healthy", "version": app.version}

# Regular API endpoints
@app.post("/api/generate-image")
async def api_generate_image(prompt: str, model: str = "Flux-Dev", step: int = 50, size: str = "640x360"):
    return await Tools.generate_image(prompt, queues, model, step, size)


# Create MCP server
mcp = FastMCP("N8N Tools")
# Register tools with MCP
@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two numbers together"""
    return Tools.add(a, b)

@mcp.tool()
async def generate_youtube_thumbnail(prompt: str) -> str:
    """Generate a youtube thumbnail corresponding to the prompt.
    
    Args:
        prompt: The prompt to generate the image
    """
    result = await Tools.text_to_image(prompt, queues, model="Flux-Dev", width=640, height=360)
    return "\n".join(result) if result else 'No image returned'


# Mount MCP server to FastAPI
app.mount("/mcp", mcp.sse_app())

if __name__ == "__main__":
    # Start server
    uvicorn.run(app, host="0.0.0.0", port=8000)
