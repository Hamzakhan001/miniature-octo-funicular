FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && apt-get install -y build-essential curl && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml ./
COPY uv.lock ./

RUN pip install --no-cache-dir uv && uv pip install --system .

COPY . .

CMD ["python", "-m", "app.workers.fargate_entrypoint"]
