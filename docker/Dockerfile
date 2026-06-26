# Dockerfile
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel

ARG DEBIAN_FRONTEND=noninteractive

ENV CONDA_DIR=/opt/conda
ENV PATH=${CONDA_DIR}/bin:${PATH}

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git wget ca-certificates \
    libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender1 \
    gdal-bin libgdal-dev libspatialindex-dev libgeos-dev libproj-dev \
    libpq-dev xvfb unzip openssh-client && \
    apt-get clean && rm -rf /var/lib/apt/lists/*


COPY environment.yml /tmp/environment.yml
COPY requirements.txt /tmp/requirements.txt

RUN conda env create -f /tmp/environment.yml -n navsim && \
    conda clean -a -y

SHELL ["conda", "run", "-n", "navsim", "/bin/bash", "-c"]
RUN pip install --no-cache-dir -r /tmp/requirements.txt

SHELL ["/bin/bash", "-c"]
RUN echo "source /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate navsim" >> ~/.bashrc

ENV CONDA_DEFAULT_ENV=navsim
ENV PATH=/opt/conda/envs/navsim/bin:$PATH

WORKDIR /workspace
CMD ["/bin/bash"]
