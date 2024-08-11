FROM python:3.10-bookworm AS builder

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1
WORKDIR /app

RUN pip install poetry
RUN poetry config virtualenvs.in-project true
COPY pyproject.toml poetry.lock ./
RUN poetry install
FROM python:3.10-slim-bookworm
WORKDIR /app
COPY --from=builder /app/.venv .venv/
COPY . .
RUN .venv/bin/pip install -r requirements.txt
CMD ["/app/.venv/bin/fastapi", "run"]
