FROM python:3.11-slim
WORKDIR /app

COPY ../../src/metrics_exporter.py ./metrics_exporter.py
RUN pip install --no-cache-dir prometheus_client

EXPOSE 8000
CMD ["python", "metrics_exporter.py"]
