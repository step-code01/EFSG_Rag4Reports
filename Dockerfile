FROM pytorch/pytorch:2.0-cuda11.8-runtime-ubuntu22.04

WORKDIR /workspace

# Install system deps
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -q -r requirements.txt

# Copy code
COPY . .

# Entry point: TIRA calls this
ENTRYPOINT ["python", "script.py"]