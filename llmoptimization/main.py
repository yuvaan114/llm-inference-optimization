import logging

from fastapi import FastAPI

from llmoptimization.config import settings

logging.basicConfig(level=settings.log_level)
logger = logging.getLogger(__name__)

app = FastAPI(title=settings.app_name, version=settings.version)


@app.get("/health")
def health():
    """Liveness probe. Returns 200 when the server process is up."""
    return {
        "status": "ok",
        "app": settings.app_name,
        "version": settings.version,
        "model": settings.model_name,   # not loaded yet — just what we intend to serve
    }


@app.get("/")
def root():
    return {"message": f"{settings.app_name} is running. See /docs."}