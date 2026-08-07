FROM python:3.10-slim

SHELL ["/bin/bash", "-c"]

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/root/.cache/huggingface \
    PATH=/opt/conda/bin:/opt/conda/envs/ragmark/bin:$PATH

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    wget \
    bzip2 \
    build-essential \
    pkg-config \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && curl -fsSL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o /tmp/miniconda.sh \
    && bash /tmp/miniconda.sh -b -p /opt/conda \
    && /opt/conda/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main \
    && /opt/conda/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r \
    && /opt/conda/bin/conda clean -afy \
    && /opt/conda/bin/conda create -y -n ragmark python=3.10 \
    && /opt/conda/bin/conda init bash \
    && echo ". /opt/conda/etc/profile.d/conda.sh" >> /root/.bashrc \
    && echo "conda activate ragmark" >> /root/.bashrc \
    && rm /tmp/miniconda.sh

COPY requirements.txt /app/requirements.txt

RUN source /opt/conda/etc/profile.d/conda.sh && conda activate ragmark && \
    pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

RUN source /opt/conda/etc/profile.d/conda.sh && conda activate ragmark && \
    conda install -y -c pytorch -c nvidia faiss-gpu=1.8.0 && \
    conda clean -afy

COPY . /app

CMD ["bash"]
