# syntax = docker/dockerfile:1
FROM condaforge/mambaforge:23.1.0-1

# install required packages with mamba
RUN --mount=type=cache,mode=0755,target=/opt/conda/pkgs \
    --mount=type=cache,mode=0755,target=/root/.cache/pip \
    --mount=type=bind,target=environment.yml,source=environment.yml \
    mamba env update -n base --file environment.yml

WORKDIR /app
COPY app app/
CMD uvicorn app.main:app --proxy-headers --host 0.0.0.0 --port 80 --reload

EXPOSE 80
