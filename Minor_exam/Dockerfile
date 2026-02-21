# ==========================================
# Base Image: NVIDIA CUDA + Ubuntu
# ==========================================
FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# ==========================================
# Install System Dependencies
# ==========================================
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    nano \
    git \
    && rm -rf /var/lib/apt/lists/*

# Make python command available
RUN ln -s /usr/bin/python3 /usr/bin/python

# Upgrade pip
RUN pip install --upgrade pip

# ==========================================
# Set Working Directory
# ==========================================
WORKDIR /app

# ==========================================
# Copy Requirements & Install Python Libraries
# ==========================================
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ==========================================
# Copy Entire Project
# ==========================================
COPY . .

# ==========================================
# Default Command: Run Evaluation
# ==========================================
CMD ["python", "src/evaluate.py"]