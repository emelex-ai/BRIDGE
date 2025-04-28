# Dockerfile-cpu.dev
#
# Base development environment for CPU-only systems
# This Dockerfile creates a foundation that other specialized environments can build upon.
# It includes all necessary development tools and Python packages, configured to run
# efficiently on CPU architectures.

FROM python:3.12-slim

# Set working directory


# Install system dependencies and development tools
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    libffi-dev \
    && rm -rf /var/lib/apt/lists/*

RUN pip install poetry==1.8.5

# Copy dependency files
COPY pyproject.toml poetry.lock* ./

RUN poetry config virtualenvs.create false

# Install dependencies
RUN poetry install --no-root --no-interaction
RUN python -m nltk.downloader cmudict
# Copy the rest of the application code
COPY . .


ENTRYPOINT ["poetry", "run", "python", "-m", "app.bin.docker_entry"]