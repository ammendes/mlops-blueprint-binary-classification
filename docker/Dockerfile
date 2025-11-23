# Multi-stage build for FastAPI Titanic Inference Service

# Builder stage
FROM python:3.10-slim AS builder
WORKDIR /app
COPY requirements.txt ./
RUN pip install --upgrade pip && pip install --user -r requirements.txt

# Runtime stage
FROM python:3.10-slim
WORKDIR /app
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH
COPY . .
EXPOSE 8000
CMD ["uvicorn", "src.inference_api:app", "--host", "0.0.0.0", "--port", "8000"]
