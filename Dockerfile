# PyTorch 2.14.1 enables deterministic dilated convolutions through cuDNN.
FROM pytorch/pytorch:2.14.1-cuda12.6-cudnn9-runtime@sha256:1262abc856a428933c0477b4be87079173a0b087b7691c47fcb396b7b5775c56

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    OMP_NUM_THREADS=4 \
    MKL_NUM_THREADS=4

# Keep project pins separate from the base image's managed Python packages.
RUN python -m venv --without-pip --system-site-packages /opt/venv
ENV PATH="/opt/venv/bin:${PATH}"

WORKDIR /workspace
COPY requirements.txt /tmp/requirements.txt
RUN python -m pip install --no-cache-dir -r /tmp/requirements.txt \
    && python -m pip check \
    && mkdir -p /tmp /var/tmp \
    && chmod 1777 /tmp /var/tmp

# Match the trainer's FP32 policy in standalone CUDA checks as well.
ENV NVIDIA_TF32_OVERRIDE=0

COPY . /workspace
CMD ["python", "train.py", "--help"]
