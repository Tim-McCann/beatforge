FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    ffmpeg \
    libsndfile1 \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Create directory for mounted files
RUN mkdir -p /mnt/files

# Install Python dependencies
RUN pip install --no-cache-dir flask

# Clone the audiocraft repository
RUN git clone https://github.com/facebookresearch/audiocraft.git /app/audiocraft
WORKDIR /app/audiocraft

# Install audiocraft
RUN pip install -e .

# Go back to app directory and copy server code
WORKDIR /app
COPY musicgen_api_server.py .

# Make sure the import paths work
ENV PYTHONPATH="/app:${PYTHONPATH}"

# Expose the port
EXPOSE 8001

# Run the server
CMD ["python", "musicgen_api_server.py"]