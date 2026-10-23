# CUDA 13.0 includes sm_120 kernels for RTX 5090; keep PyTorch 2.14.1 fixed.
FROM pytorch/pytorch:2.14.1-cuda13.0-cudnn9-runtime@sha256:3c0b174a1339193a5bcc4b37d737c9f5ca3c4782a2293d3888e4271367eddf44

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
