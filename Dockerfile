FROM docker:27-cli AS dockercli
FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PIP_NO_CACHE_DIR=1 HF_HUB_DISABLE_XET=1
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends tesseract-ocr libgomp1 \
    && rm -rf /var/lib/apt/lists/*
COPY --from=dockercli /usr/local/bin/docker /usr/local/bin/docker
RUN pip install uv==0.9.30
COPY pyproject.toml uv.lock ./
COPY assessment ./assessment
RUN uv sync --frozen --no-dev
ENV PATH="/app/.venv/bin:$PATH"
COPY app.py chat.py ./
COPY .streamlit ./.streamlit
COPY scripts ./scripts
COPY configs ./configs
COPY evals ./evals
COPY docs ./docs
CMD ["uvicorn", "assessment.api:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
