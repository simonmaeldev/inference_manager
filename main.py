
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from mcp.server.fastmcp import FastMCP
import uvicorn

from src.routes.tools import Tools


# Create FastAPI app
app = FastAPI(
    title="LocalAI Inference Manager",
    description="Queue-based inference server for LocalAI",
    version="0.1.0",
    docs_url="/docs"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create MCP server
mcp = FastMCP("N8N Tools")


@app.get("/health")
async def health_check():
    """Basic health check endpoint"""
    return {"status": "healthy", "version": app.version}

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
    result = await Tools.text_to_image(prompt, model="Flux-Dev", width=640, height=360)
    return f"Image generated: {result[0] if result else 'No image returned'}"

# Register FastAPI endpoints
@app.get("/add")
def api_add(a: int, b: int):
    return {"result": Tools.add(a, b)}

@app.post("/generate-image")
async def api_generate_image(prompt: str, model: str = "flux.1-dev", step: int = 50, size: str = "640x360"):
    return await Tools.generate_image(prompt, model, step, size)

@mcp.tool()
async def generate_text(messages: list, model: str = "gpt-4", temperature: float = 0.7) -> dict:
    """Generate text using the local API service.
    
    Args:
        messages: List of chat messages with role and content
        model: The model to use for generation
        temperature: Sampling temperature
    """
    result = await Tools.generate_text(messages, model, temperature)
    return f"Text generated: {result.get('choices', [{}])[0].get('message', {}).get('content', 'No content returned')}"

@app.post("/generate-text")
async def api_generate_text(messages: list, model: str = "gpt-4", temperature: float = 0.7):
    return await Tools.generate_text(messages, model, temperature)

# Mount MCP server to FastAPI
app.mount("/mcp", mcp.sse_app())

async def process_queue():
    """Process all requests in priority order"""
    while True:
        for model_type in queues.priority_order:
            for model in queues.model_sets[model_type]:
                while queues.queues[model]:
                    request = queues.queues[model].popleft()
                    try:
                        await request.process()
                        request.future.set_result(None)
                    except Exception as e:
                        request.future.set_exception(e)

if __name__ == "__main__":
    import asyncio
    from src.models.queues import InferenceQueues
    queues = InferenceQueues()
    
    # Start queue processing in background
    asyncio.create_task(process_queue())
    
    # Start server
    uvicorn.run(app, host="0.0.0.0", port=8000)
