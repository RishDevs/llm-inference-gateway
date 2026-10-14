FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    HF_HOME=/models

WORKDIR /app
COPY pyproject.toml README.md LICENSE ./
COPY gateway ./gateway
COPY bench ./bench
RUN pip install --upgrade pip && pip install ".[bench]"

EXPOSE 8000
VOLUME ["/models"]
ENTRYPOINT ["python", "-m", "gateway.server"]
CMD ["--preset", "gateway"]
