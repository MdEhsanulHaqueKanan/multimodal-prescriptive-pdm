# Stage 1: The "Builder" Stage
FROM python:3.10 as builder
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y --no-install-recommends git git-lfs && \
    git lfs install
COPY requirements.txt .
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --no-cache-dir -r requirements.txt && pip install gevent
# Download the embedding model during the build ---
COPY scripts/download_embedding_model.py .
RUN python download_embedding_model.py

# Stage 2: The Final "Production" Stage
FROM python:3.10
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends git git-lfs && \
    git lfs install
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# --- NEW LINE TO ADD ---
# Copy the pre-downloaded embedding model from the builder stage
COPY --from=builder /app/embedding_model ./embedding_model

# Copy the rest of the application code
COPY . .
RUN git lfs pull
ENV LLM_MODEL_NAME="phi-3:mini"
EXPOSE 10000
CMD ["gunicorn", "--worker-class", "gevent", "--bind", "0.0.0.0:10000", "--timeout", "300", "run:app"]