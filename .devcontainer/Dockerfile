# Multi-arch base image (linux/amd64 + linux/arm64): works on Windows, Linux,
# Intel Macs and Apple Silicon Macs.
FROM mcr.microsoft.com/devcontainers/python:1-3.11-bookworm

# System libs:
#  - ffmpeg            -> read/seek the tactical & broadcast .mp4 (opencv / imageio)
#  - libgl1, libglib2  -> runtime deps of opencv
#  - libxml2/libxslt   -> XML parsing (ATD metadata, Smart Tagging)
#  - build-essential   -> fallback compile for any package lacking an arm64 wheel
# The base image ships a Yarn apt repo whose signing key has expired, which
# breaks `apt-get update`; we don't need Yarn, so drop that source first.
RUN rm -f /etc/apt/sources.list.d/yarn.list \
    && apt-get update && export DEBIAN_FRONTEND=noninteractive \
    && apt-get install -y --no-install-recommends \
        ffmpeg \
        libgl1 \
        libglib2.0-0 \
        libxml2-dev \
        libxslt1-dev \
        build-essential \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH

# Default virtual environment, baked into the image (not the bind-mounted
# workspace) so it never clashes with a host venv and stays fast on Win/macOS.
COPY requirements.txt /tmp/requirements.txt
RUN python -m venv /opt/venv \
    && /opt/venv/bin/pip install --no-cache-dir --upgrade pip setuptools wheel \
    && /opt/venv/bin/pip install --no-cache-dir -r /tmp/requirements.txt \
    && /opt/venv/bin/python -m ipykernel install --prefix /opt/venv --name mtah --display-name "Python (MTAH venv)" \
    && chown -R vscode:vscode /opt/venv

USER vscode
