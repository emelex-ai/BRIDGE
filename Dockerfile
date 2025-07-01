FROM python:3.12-slim

WORKDIR /app

# Install necessary build tools
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy application code
COPY bridge/ /app/bridge/
COPY app/ /app/app/
COPY data/tests/ /app/data/tests/
COPY tests/application/training/data/ /app/tests/application/training/data/
COPY pyproject.toml uv.lock ./

# Create directories first
RUN mkdir -p /app/data /app/model_artifacts /app/results

# Copy data files 
COPY data/phonreps.csv /app/data/

# Install uv and dependencies
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
RUN uv sync --frozen --no-install-project --no-dev

# Download NLTK data
RUN uv run python -c "import nltk; nltk.download('cmudict')"

# Set environment variables for memory optimization
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV PYTHONOPTIMIZE=2

# Set entry point
ENTRYPOINT ["./venv/bin/python", "-m", "app.bin.docker_entry"]
