# Multi-stage build for MDLM project
# Stage 1: Base image with CUDA and Python
FROM nvidia/cuda:12.2.2-runtime-ubuntu22.04 AS base

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    CUDA_HOME=/usr/local/cuda \
    PATH=/usr/local/cuda/bin:${PATH}

# Install Python 3.11 and dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-venv \
    python3-pip \
    git \
    wget \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/* \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

# Create working directory
WORKDIR /app

# Stage 2: Build dependencies
FROM base AS builder

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN python3.11 -m pip install --upgrade pip && \
    python3.11 -m pip install --user --no-warn-script-location -r requirements.txt

# Stage 3: Runtime image
FROM base AS runtime

# Copy Python packages from builder
COPY --from=builder /root/.local /root/.local

# Set PATH to include user site-packages
ENV PATH=/root/.local/bin:${PATH} \
    PYTHONPATH=/root/.local/lib/python3.11/site-packages:${PYTHONPATH}

# Copy application code
COPY . /app

# Create directories for checkpoints and cache
RUN mkdir -p /app/.checkpoint /app/results

# Set default command
ENTRYPOINT ["python3.11"]
CMD ["full.py"]

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python3.11 -c "import torch; print('CUDA available:', torch.cuda.is_available())" || exit 1
