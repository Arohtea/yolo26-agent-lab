FROM python:3.12-slim-bookworm
ARG TORCH_FLAVOR=cpu
WORKDIR /app
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates libcurl4 libgomp1 \
    && rm -rf /var/lib/apt/lists/*
COPY pyproject.toml ./
COPY src ./src
COPY scripts ./scripts
COPY docs/skills ./docs/skills
RUN pip install --no-cache-dir . \
    && python scripts/install_torch.py --flavor ${TORCH_FLAVOR} --allow-system-python \
    && pip install --no-cache-dir '.[training,evaluation,agent,export]'
COPY configs ./configs
COPY migrations ./migrations
COPY alembic.ini ./
COPY docs/*.md ./docs/
COPY docs/vendor ./docs/vendor
ENV YOLO_STORAGE_DIR=/app/storage PYTHONUNBUFFERED=1
CMD ["yolo-agent", "api"]
