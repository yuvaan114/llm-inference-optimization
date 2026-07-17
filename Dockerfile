FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# Install CPU-only PyTorch. Long timeout + retries to survive a slow connection.
RUN pip install --no-cache-dir --timeout 600 --retries 10 \
    torch --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir --timeout 600 --retries 10 -r requirements.txt

COPY llmoptimization ./llmoptimization

EXPOSE 8000

CMD ["uvicorn", "llmoptimization.main:app", "--host", "0.0.0.0", "--port", "8000"]