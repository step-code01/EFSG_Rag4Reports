FROM python:3.10-slim

WORKDIR /workspace

RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -q -r requirements.txt

COPY script.py .

ENTRYPOINT ["python", "script.py"]