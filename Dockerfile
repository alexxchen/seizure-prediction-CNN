# CUDA 12.4 and cuDNN 9 support the NVIDIA A100.
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime@sha256:c8268a92a69bd500f8be0e665b2630ee006dadaf7bfbc24249141b15ff622755 AS dependencies

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MPLBACKEND=Agg \
    OMP_NUM_THREADS=4 \
    MKL_NUM_THREADS=4

WORKDIR /workspace
COPY requirements.txt requirements-docker.txt requirements-dev.txt /tmp/
RUN python -m pip install --no-cache-dir -r /tmp/requirements-docker.txt -r /tmp/requirements-dev.txt \
    && python -m pip check

FROM dependencies AS runtime
COPY . /workspace
RUN mkdir -p /tmp /var/tmp \
    && chmod 1777 /tmp /var/tmp \
    && python docker/configure.py \
    && python -m compileall -q main.py models myio utils \
    && python -m pytest -q tests \
    && python docker/smoke_test.py --device cpu

CMD ["python", "main.py", "--mode", "cv", "--dataset", "CHBMIT", "--targets", "20", "--device", "auto"]
