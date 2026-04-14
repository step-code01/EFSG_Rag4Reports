FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04

WORKDIR /workspace

# Install Python and system deps
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    git \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --upgrade pip

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -q -r requirements.txt

# Copy code
COPY . .

# Entry point
ENTRYPOINT ["python3", "script.py"]