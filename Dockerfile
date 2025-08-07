# Stage 1: The "Builder" Stage
FROM python:3.10 as builder
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 git git-lfs curl && \
    git lfs install
COPY requirements.txt .
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --no-cache-dir -r requirements.txt
COPY scripts/download_embedding_model.py .
RUN python download_embedding_model.py

# Stage 2: The Final "Production" Stage
FROM python:3.10
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 git git-lfs curl
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
COPY --from=builder /app/embedding_model ./embedding_model
COPY . .
RUN git lfs pull

# Install Ollama, start the server in the background, pull the model, and then stop the server.
RUN curl -fsSL https://ollama.com/install.sh | sh && \
    /bin/bash -c "ollama serve &" && \
    sleep 5 && \
    ollama pull gemma:2b && \
    pkill ollama

# Expose the application port
EXPOSE 7860

# Start Ollama in the background, wait for it to be ready, then start Gunicorn
CMD ["/bin/bash", "-c", "ollama serve & sleep 10 && gunicorn --bind 0.0.0.0:7860 --timeout 300 run:app"]