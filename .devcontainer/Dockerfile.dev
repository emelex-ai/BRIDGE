FROM python:3.10

WORKDIR /app

# Install basic build dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential libffi-dev git && \
    rm -rf /var/lib/apt/lists/*

# Install poetry
RUN pip install poetry

# Copy poetry files and install dependencies
COPY poetry.lock pyproject.toml /app/
RUN poetry config virtualenvs.create false \
    && poetry install --no-dev

# Copy the rest of the application
COPY . /app


