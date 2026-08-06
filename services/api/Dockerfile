FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.11.15 /uv /usr/local/bin/uv

WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN uv sync --no-dev --frozen

COPY app ./app
COPY services/api ./services/api

ENTRYPOINT ["uv", "run", "--no-sync", "uvicorn", "services.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
