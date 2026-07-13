# LLM Inference Server

Production-style LLM serving stack, built for the Viettel AI Race
LLM Inference Optimization challenge. Developed CPU-first against a
small stand-in model, then moved to NVIDIA H200 + vLLM.

## Run locally
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn llmoptimization.main:app --reload --port 8000

## Check it's alive
curl http://localhost:8000/health