FROM python:3.12-slim

WORKDIR /app

# Install necessary build tools and curl for uv installation
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.cargo/bin:$PATH"

# Copy pyproject.toml and uv.lock first for better caching
COPY pyproject.toml uv.lock* ./

# Install dependencies using uv
RUN uv sync --frozen --no-dev

# Copy application code
COPY bridge/ /app/bridge/
COPY app/ /app/app/
COPY data/tests/ /app/data/tests/
COPY tests/application/training/data/ /app/tests/application/training/data/

# Create directories first
RUN mkdir -p /app/data /app/model_artifacts /app/results

# Copy data files
COPY data/phonreps.csv /app/data/

# Download NLTK data using uv run
RUN uv run python -m nltk.downloader cmudict

# Set environment variables for memory optimization
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV PYTHONOPTIMIZE=2

# Set entry point using uv run
ENTRYPOINT ["uv", "run", "python", "-m", "app.bin.docker_entry"]
