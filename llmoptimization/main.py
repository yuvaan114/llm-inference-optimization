import json
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from llmoptimization.config import settings
from llmoptimization.engine import engine

logging.basicConfig(level=settings.log_level)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    engine.load()
    yield


app = FastAPI(title=settings.app_name, version=settings.version, lifespan=lifespan)


class GenerateRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 128
    temperature: float = 0.7


@app.get("/health")
def health():
    return {
        "status": "ok",
        "app": settings.app_name,
        "version": settings.version,
        "model": settings.model_name,
        "model_loaded": engine.ready,
    }


@app.post("/generate")
def generate(req: GenerateRequest):
    return engine.generate(req.prompt, req.max_new_tokens, req.temperature)


@app.post("/generate/stream")
def generate_stream(req: GenerateRequest):
    def event_stream():
        for event in engine.generate_stream(req.prompt, req.max_new_tokens, req.temperature):
            yield f"data: {json.dumps(event)}\n\n"
    return StreamingResponse(event_stream(), media_type="text/event-stream")

@app.get("/")
def root():
    return {"message": f"{settings.app_name} is running. See /docs."}