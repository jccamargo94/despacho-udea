FROM python:3.12-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends coinor-cbc \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.11.15 /uv /usr/local/bin/uv

WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN uv sync --no-dev --frozen

COPY app ./app
COPY tests/fixtures/xm_smoke ./tests/fixtures/xm_smoke

ENTRYPOINT ["uv", "run", "--no-sync", "python", "-m", "app"]
CMD ["run"]
