FROM python:3.12-slim

WORKDIR /app

# Install necessary build tools
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy application code
COPY . /app/


# Create directories first
RUN mkdir -p /app/data /app/model_artifacts /app/results

# Copy data files 
COPY data/phonreps.csv /app/data/

# Install Poetry and dependencies (excluding GPU)
RUN pip install poetry==1.8.5 && \
    poetry config virtualenvs.create false && \
    poetry install

# Download NLTK data
RUN python -m nltk.downloader cmudict

# Set environment variables for memory optimization
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV PYTHONOPTIMIZE=2

# Set entry point
ENTRYPOINT ["python", "-m", "app.bin.finetuning_docker_entry"]