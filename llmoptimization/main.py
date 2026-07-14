import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from pydantic import BaseModel

from llmoptimization.config import settings
from llmoptimization.engine import engine

logging.basicConfig(level=settings.log_level)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    engine.load()      # runs once, when the server starts
    yield              # server handles requests here
    # (shutdown cleanup would go after yield)


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
        "model_loaded": engine.ready,   # now reports real load state
    }


@app.post("/generate")
def generate(req: GenerateRequest):
    return engine.generate(req.prompt, req.max_new_tokens, req.temperature)


@app.get("/")
def root():
    return {"message": f"{settings.app_name} is running. See /docs."}