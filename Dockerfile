# CUDA 12.4 and cuDNN 9 support the NVIDIA A100.
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime@sha256:c8268a92a69bd500f8be0e665b2630ee006dadaf7bfbc24249141b15ff622755

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    OMP_NUM_THREADS=4 \
    MKL_NUM_THREADS=4

WORKDIR /workspace
COPY requirements.txt /tmp/requirements.txt
# Replace the base image's incompatible ninja wheel metadata.
RUN python -m pip install --no-cache-dir -r /tmp/requirements.txt ninja==1.11.1.3 \
    && python -m pip check \
    && mkdir -p /tmp /var/tmp \
    && chmod 1777 /tmp /var/tmp

COPY . /workspace
CMD ["python", "train.py", "--help"]
