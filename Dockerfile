FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV CONDA_DIR=/opt/conda
ENV PATH=/opt/conda/envs/drivesuprim/bin:/opt/conda/bin:$PATH
ENV NAVSIM_DEVKIT_ROOT=/workspace/DriveSuprim

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl git build-essential \
    libglib2.0-0 libgl1 libgomp1 libsm6 libxext6 libxrender1 \
    libgeos-dev libgdal-dev libproj-dev libspatialindex-dev \
    && rm -rf /var/lib/apt/lists/*

RUN curl -fsSL \
    https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh \
    -o /tmp/miniforge.sh \
    && bash /tmp/miniforge.sh -b -p "${CONDA_DIR}" \
    && rm /tmp/miniforge.sh \
    && conda create -y -n drivesuprim python=3.9 pip \
    && conda clean -afy

WORKDIR /workspace/DriveSuprim
COPY . /workspace/DriveSuprim

RUN python -m pip install --upgrade pip \
    && python -m pip install torch==2.0.1 \
       --index-url https://download.pytorch.org/whl/cu118 \
    && sed '/^[[:space:]]*xformers[[:space:]]*$/d' requirements.txt \
       > /tmp/requirements.txt \
    && python -m pip install -r /tmp/requirements.txt \
    && python -m pip install xformers==0.0.21 --no-deps \
    && python -m mim install mmcv-full==1.7.2 mmdet==2.28.2 mmengine \
    && python -m pip install -e .

RUN mkdir -p /tmp/ray

CMD ["bash"]


RUN python -m pip install --no-cache-dir --force-reinstall --no-deps \
    torchvision==0.15.2 \
    --index-url https://download.pytorch.org/whl/cu118
