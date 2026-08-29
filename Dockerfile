FROM python:3.12-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN uv venv /opt/venv \
    && uv pip install --python /opt/venv/bin/python --no-cache -r /app/requirements.txt

FROM python:3.12-slim AS runtime

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1

WORKDIR /app

COPY --from=builder /opt/venv /opt/venv

COPY src /app/src
COPY app /app/app
COPY artifacts/cats_vs_dogs_cnn.pt /app/artifacts/cats_vs_dogs_cnn.pt

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]