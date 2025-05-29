FROM python:3.9

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    ffmpeg \
    libsndfile1 \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

COPY musicgen_api_server.py .

# Install PyTorch 2.2.0 with CUDA 12.1 support (includes get_default_device)
RUN pip install 'torch==2.1.0'

# Install other Python dependencies
RUN pip install setuptools wheel
RUN git clone https://github.com/facebookresearch/audiocraft.git
RUN pip install -e ./audiocraft
RUN pip uninstall transformers -y
RUN pip install transformers==4.33.3
RUN pip install flask

# Make sure the import paths work
ENV PYTHONPATH="/app:/app/audiocraft:${PYTHONPATH}"

# Expose the port
EXPOSE 8001

# Run the server
CMD ["python", "musicgen_api_server.py"]