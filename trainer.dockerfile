# Base image
FROM python:3.9-slim


# install python
RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*
COPY requirements.txt requirements.txt
COPY pyproject.toml pyproject.toml
COPY s3_reproducibility/src/ src/
COPY data/ data/
WORKDIR /

# Syntax for Docker >= 18.09
# Mount pip cache directory to enable caching
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.txt

RUN pip install . --no-deps --no-cache-dir

# Ensure with -u that prints are redirected to the overlaying OS term and not contained in the docker logs
ENTRYPOINT ["python", "-u", "src/train_model.py", "train"] 