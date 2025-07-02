FROM python:3.12-slim

WORKDIR /app

# Install necessary build tools
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /app/data /app/model_artifacts /app/results
COPY  pyproject.toml /app/
    
# Install Poetry and dependencies (excluding GPU)
RUN pip install poetry==1.8.5 && \
    poetry config virtualenvs.create false && \
    poetry install
# Copy application code
COPY bridge/ /app/bridge/
COPY app/ /app/app/
COPY data/tests/ /app/data/tests/
COPY tests/application/training/data/ /app/tests/application/training/data/
COPY pyproject.toml poetry.lock* ./

# Create directories first

# Copy data files 
COPY data/phonreps.csv /app/data/


# Download NLTK data
RUN python -m nltk.downloader cmudict

# Copy application code
COPY . /app/

# Set environment variables for memory optimization
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV PYTHONOPTIMIZE=2

# Set entry point
ENTRYPOINT ["python", "-m", "app.bin.finetuning_docker_entry"]