FROM mcr.microsoft.com/devcontainers/python:3.0.6-3.14-trixie

RUN apt-get update && apt-get install -y \
    tig

RUN pip install --no-cache-dir \
    jupyter==1.1.1 \
    pytest==9.0.2
