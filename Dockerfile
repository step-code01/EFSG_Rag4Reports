FROM python:3.10-slim

WORKDIR /workspace

RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -q -r requirements.txt

COPY script.py .

ENV GROQ_API_KEY=""
ENV HF_TOKEN=""

ENTRYPOINT ["python", "script.py"]