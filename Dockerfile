FROM python:3.12-slim

# Minihack requires git.
RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .

# Install CPU-only PyTorch first to avoid installing the wrong version of torch and torchvision.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Install the rest of the requirements, including minihack.
RUN pip install --no-cache-dir -r requirements.txt